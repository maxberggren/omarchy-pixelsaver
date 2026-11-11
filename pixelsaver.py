#!/usr/bin/env python3
"""PIXELSAVER - a pixel-art / demoscene terminal screensaver.

Reads the Omarchy screensaver art (~/.config/omarchy/branding/screensaver.txt),
converts its block characters into a real pixel bitmap, and renders animated
scenes at "half-block" resolution (every terminal cell = 2 square pixels).

Live:        ./pixelsaver.py                     (n/p = next/prev scene, s = screenshot, any other key quits)
One scene:   ./pixelsaver.py --scene fire
Screenshot:  ./pixelsaver.py --shot fire --at 6 --out fire.png
Sheet:       ./pixelsaver.py --sheet sheet.png --times 2,6,12
GIF:         ./pixelsaver.py --gif warp --seconds 6 --out warp.gif
"""
import argparse
import heapq
import math
import os
import select
import signal
import subprocess
import sys
import time

import numpy as np

DEFAULT_LOGO = os.path.expanduser("~/.config/omarchy/branding/screensaver.txt")
FPS = 30
DT = 1.0 / FPS

# --------------------------------------------------------------------------- helpers


def grad(stops, n=256):
    xs = np.array([s[0] for s in stops], float)
    cs = np.array([s[1] for s in stops], float)
    x = np.linspace(0, 1, n)
    return np.stack([np.interp(x, xs, cs[:, i]) for i in range(3)], -1)


def hsv2rgb(h, s, v):
    h = np.asarray(h, float)
    s = np.broadcast_to(np.asarray(s, float), h.shape)
    v = np.broadcast_to(np.asarray(v, float), h.shape)
    h6 = (h % 1.0) * 6.0
    i = np.floor(h6).astype(int) % 6
    f = h6 - np.floor(h6)
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return np.stack([r, g, b], -1) * 255.0


def shift(a, dy, dx):
    out = np.zeros_like(a)
    H, W = a.shape[:2]
    if abs(dy) >= H or abs(dx) >= W:
        return out
    ys, yd = (slice(0, H - dy), slice(dy, H)) if dy >= 0 else (slice(-dy, H), slice(0, H + dy))
    xs, xd = (slice(0, W - dx), slice(dx, W)) if dx >= 0 else (slice(-dx, W), slice(0, W + dx))
    out[yd, xd] = a[ys, xs]
    return out


def dilate(m, r=1, diag=True):
    out = m.copy()
    for _ in range(r):
        cur = out.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if (dy or dx) and (diag or not (dy and dx)):
                    out |= shift(cur, dy, dx)
    return out


def box_blur(a, r):
    if r <= 0:
        return a
    a = a.astype(float)
    k = 2 * r + 1
    for axis in (0, 1):
        pad = [(0, 0)] * a.ndim
        pad[axis] = (r + 1, r)
        c = np.cumsum(np.pad(a, pad, mode="edge"), axis=axis)
        n = a.shape[axis]
        a = (np.take(c, np.arange(k, k + n), axis=axis) - np.take(c, np.arange(0, n), axis=axis)) / k
    return a


def smoothstep(x, a, b):
    x = np.clip((x - a) / (b - a), 0, 1)
    return x * x * (3 - 2 * x)


B4 = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], float)


def bayer8():
    b = B4
    return np.block([[4 * b + 0, 4 * b + 2], [4 * b + 3, 4 * b + 1]]) / 64.0


def tile(pat, H, W):
    return np.tile(pat, (H // pat.shape[0] + 1, W // pat.shape[1] + 1))[:H, :W]


def blit(img, mask, x, y, color):
    H, W = img.shape[:2]
    h, w = mask.shape
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x0 >= x1 or y0 >= y1:
        return
    sub = mask[y0 - y:y1 - y, x0 - x:x1 - x]
    img[y0:y1, x0:x1][sub] = color


# --------------------------------------------------------------------------- 3x5 pixel font

_FONT_SRC = {
    "A": ".#./#.#/###/#.#/#.#", "B": "##./#.#/##./#.#/##.", "C": ".##/#../#../#../.##",
    "D": "##./#.#/#.#/#.#/##.", "E": "###/#../##./#../###", "F": "###/#../##./#../#..",
    "G": ".##/#../#.#/#.#/.##", "H": "#.#/#.#/###/#.#/#.#", "I": "###/.#./.#./.#./###",
    "J": "..#/..#/..#/#.#/.#.", "K": "#.#/#.#/##./#.#/#.#", "L": "#../#../#../#../###",
    "M": "#.#/###/###/#.#/#.#", "N": "##./#.#/#.#/#.#/#.#", "O": ".#./#.#/#.#/#.#/.#.",
    "P": "##./#.#/##./#../#..", "Q": ".#./#.#/#.#/##./.##", "R": "##./#.#/##./#.#/#.#",
    "S": ".##/#../.#./..#/##.", "T": "###/.#./.#./.#./.#.", "U": "#.#/#.#/#.#/#.#/###",
    "V": "#.#/#.#/#.#/#.#/.#.", "W": "#.#/#.#/###/###/#.#", "X": "#.#/#.#/.#./#.#/#.#",
    "Y": "#.#/#.#/.#./.#./.#.", "Z": "###/..#/.#./#../###",
    "0": "###/#.#/#.#/#.#/###", "1": ".#./##./.#./.#./###", "2": "##./..#/.#./#../###",
    "3": "##./..#/.#./..#/##.", "4": "#.#/#.#/###/..#/..#", "5": "###/#../##./..#/##.",
    "6": ".##/#../###/#.#/###", "7": "###/..#/.#./.#./.#.", "8": "###/#.#/###/#.#/###",
    "9": "###/#.#/###/..#/##.",
    " ": ".../.../.../.../...", ">": "#../.#./..#/.#./#..", "<": "..#/.#./#../.#./..#",
    "_": ".../.../.../.../###", ":": ".../.#./.../.#./...", ".": ".../.../.../.../.#.",
    "-": ".../.../###/.../...", "!": ".#./.#./.#./.../.#.", "/": "..#/..#/.#./#../#..",
    "*": "#.#/.#./###/.#./#.#", "=": ".../###/.../###/...", "+": ".../.#./###/.#./...",
    "[": "##./#../#../#../##.", "]": ".##/..#/..#/..#/.##", "%": "#.#/..#/.#./#../#.#",
    ",": ".../.../.../.#./#..", "?": "##./..#/.#./.../.#.", "'": ".#./.#./.../.../...",
    "#": "#.#/###/#.#/###/#.#", '"': "#.#/#.#/.../.../...", "&": ".#./#.#/.#./#.#/.##", "♥": "#.#/###/###/.#./...", "█": "###/###/###/###/###",
}
FONT = {k: np.array([[c == "#" for c in row] for row in v.split("/")]) for k, v in _FONT_SRC.items()}


def text_mask(s, scale=1):
    parts = []
    for ch in s.upper():
        parts.append(FONT.get(ch, FONT["?"]))
        parts.append(np.zeros((5, 1), bool))
    m = np.hstack(parts[:-1]) if parts else np.zeros((5, 0), bool)
    if scale > 1:
        m = np.kron(m, np.ones((scale, scale), bool))
    return m


def draw_text(img, s, x, y, color, scale=1, shadow=None):
    m = text_mask(s, scale)
    if shadow is not None:
        blit(img, m, x + scale, y + scale, shadow)
    blit(img, m, x, y, color)
    return m.shape[1]


# --------------------------------------------------------------------------- logo

_BLOCKS = {"▀": (1, 0), "▄": (0, 1), "█": (1, 1), "▌": (1, 1), "▐": (1, 1),
           "░": (1, 1), "▒": (1, 1), "▓": (1, 1), "▘": (1, 0), "▝": (1, 0), "▖": (0, 1), "▗": (0, 1)}


def load_logo(path):
    try:
        lines = open(path, encoding="utf-8").read().split("\n")
    except OSError:
        lines = ["PIXEL", "SAVER"]
    w = max((len(l) for l in lines), default=1)
    m = np.zeros((len(lines) * 2, max(w, 1)), bool)
    for r, line in enumerate(lines):
        for c, ch in enumerate(line):
            if ch.isspace():
                continue
            top, bot = _BLOCKS.get(ch, (1, 1))
            m[2 * r, c], m[2 * r + 1, c] = top, bot
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return text_mask("PIXELSAVER", 2)
    return m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


# --------------------------------------------------------------------------- particles


class Particles:
    def __init__(self):
        self.p = np.zeros((0, 2))
        self.v = np.zeros((0, 2))
        self.c = np.zeros((0, 3))
        self.l = np.zeros(0)
        self.m = np.zeros(0)

    def add(self, p, v, c, life):
        p = np.atleast_2d(np.asarray(p, float))
        n = len(p)
        if n == 0:
            return
        self.p = np.vstack([self.p, p])
        self.v = np.vstack([self.v, np.broadcast_to(np.asarray(v, float), (n, 2))])
        self.c = np.vstack([self.c, np.broadcast_to(np.asarray(c, float), (n, 3))])
        life = np.broadcast_to(np.asarray(life, float), (n,))
        self.l = np.concatenate([self.l, life])
        self.m = np.concatenate([self.m, life])

    def update(self, dt, g=0.0, drag=0.0):
        if not len(self.l):
            return
        self.v[:, 1] += g * dt
        if drag:
            self.v *= math.exp(-drag * dt)
        self.p += self.v * dt
        self.l -= dt
        k = self.l > 0
        self.p, self.v, self.c, self.l, self.m = self.p[k], self.v[k], self.c[k], self.l[k], self.m[k]

    def draw(self, img, add=False):
        if not len(self.l):
            return
        H, W = img.shape[:2]
        ix = np.round(self.p[:, 0]).astype(int)
        iy = np.round(self.p[:, 1]).astype(int)
        ok = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
        a = np.clip(self.l / self.m, 0, 1)[ok, None]
        ix, iy = ix[ok], iy[ok]
        if add:
            np.add.at(img, (iy, ix), self.c[ok] * a)
        else:
            img[iy, ix] = img[iy, ix] * (1 - a) + self.c[ok] * a


# --------------------------------------------------------------------------- scene base


class Scene:
    name = "base"
    title = "BASE"

    def __init__(self, W, H, logo, seed=0):
        self.W, self.H = W, H
        self.rng = np.random.default_rng(seed)
        self.t = 0.0
        self.logo_src = logo
        yy, xx = np.mgrid[0:H, 0:W]
        self.yy, self.xx = yy.astype(float), xx.astype(float)
        self.ts = max(1, H // 64)  # HUD text scale
        self.place()
        self.init()

    def place(self, maxw=0.86, maxh=0.5, cy=0.5):
        lh, lw = self.logo_src.shape
        s = max(1, int(min(self.W * maxw / lw, self.H * maxh / lh)))
        m = np.kron(self.logo_src, np.ones((s, s), bool)) if s > 1 else self.logo_src.copy()
        h, w = m.shape
        x, y = (self.W - w) // 2, int(self.H * cy - h / 2)
        L = np.zeros((self.H, self.W), bool)
        blit(L, m, x, y, True)
        self.s, self.L = s, L
        self.lx, self.ly, self.lw, self.lh = x, y, w, h
        self.lcx, self.lcy = x + w / 2, y + h / 2

    def init(self):
        pass

    def draw(self, dt):
        raise NotImplementedError

    def frame(self, dt):
        self.t += dt
        return np.clip(self.draw(dt), 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- shared bits


# --------------------------------------------------------------------------- 5. SIGNAL CORRUPT (glitch art)

BARS = [(192, 192, 192), (192, 192, 0), (0, 192, 192), (0, 192, 0), (192, 0, 192), (192, 0, 0), (0, 0, 192)]


class Glitch(Scene):
    name, title = "glitch", "SIGNAL CORRUPT"
    CYCLE = 8.0

    def init(self):
        W, H = self.W, self.H
        self.place(maxw=0.8, maxh=0.5)
        self.bay = tile(np.kron(B4 / 16, np.ones((2, 2))), H, W)
        self.bg = np.array([(5, 3, 16), (20, 8, 48), (42, 20, 96), (70, 50, 150), (110, 100, 200)], float)
        self.fg = grad([(0, (255, 0, 140)), (0.25, (255, 190, 0)), (0.5, (0, 255, 190)),
                        (0.75, (100, 40, 255)), (1, (255, 0, 140))])
        d = dilate(self.L, 1)
        self.out = d & ~self.L
        self.shadow = shift(d, 2 * self.s, 2 * self.s) & ~d
        self.zone = dilate(self.L, 3)
        x2, y2 = (self.xx // 2) * 2, (self.yy // 2) * 2
        self.x2, self.y2 = x2, y2
        self.dist2 = np.hypot(x2 - W / 2, y2 - H / 2)
        self.next_g, self.g_end, self.kind, self.msg = 1.0, 0.0, None, None
        self.prev = None

    def micro(self, img, k):
        r, W, H = self.rng, self.W, self.H
        for _ in range(int(r.integers(2, 4 + int(10 * k)))):
            y0 = int(r.integers(0, H))
            h = int(r.integers(1, max(2, H // 10)))
            img[y0:y0 + h] = np.roll(img[y0:y0 + h], int(r.choice([-1, 1]) * r.integers(2, 4 + int(W * 0.25 * k))), 1)
        sp = int(2 + 6 * k)
        img[..., 0] = np.roll(img[..., 0], sp, 1)
        img[..., 2] = np.roll(img[..., 2], -sp, 1)
        if self.prev is not None:
            for _ in range(int(r.integers(1, 3 + int(6 * k)))):
                bw, bh = int(r.integers(4, max(5, W // 4))), int(r.integers(2, max(3, H // 5)))
                bx, by = int(r.integers(0, W - bw)), int(r.integers(0, H - bh))
                P = np.roll(self.prev, (int(r.integers(-4, 5)), int(r.integers(-8, 9))), (0, 1))
                img[by:by + bh, bx:bx + bw] = P[by:by + bh, bx:bx + bw]
        return img

    def draw(self, dt):
        W, H, r, t = self.W, self.H, self.rng, self.t
        ph = t % self.CYCLE
        u, v = self.x2 * 0.09, self.y2 * 0.09
        p = (np.sin(u + t) + np.sin(v * 1.3 + t * 1.1) + np.sin((u + v) * 0.7 + t * 0.6)
             + np.sin(self.dist2 * 0.14 - t * 1.8))
        p = (p + 4) / 8
        idx = np.clip(np.floor(p * 4 + self.bay - 0.5), 0, 4).astype(int)
        img = self.bg[idx].copy()
        img[self.shadow] = (0, 0, 0)
        img[self.out] = (0, 0, 0)
        fc = self.fg[((p * 1.6 + t * 0.12 + self.xx * 0.004) % 1 * 255).astype(int)]
        img[self.L] = fc[self.L]
        ab = img.copy()
        ab[..., 0] = np.roll(img[..., 0], self.s, 1)
        ab[..., 2] = np.roll(img[..., 2], -self.s, 1)
        img[self.zone] = ab[self.zone]

        ts = self.ts
        if ph < 5.2:
            if t > self.next_g:
                self.g_end = t + r.uniform(0.15, 0.45)
                self.next_g = self.g_end + r.uniform(0.5, 1.4)
                self.msg = (str(r.choice(["ERR 0XC0FFEE", "SEGFAULT", "NO CARRIER", "0XDEADBEEF", "CORE DUMPED",
                                          "STACK SMASHED", "EOF", "PARITY ERR"])),
                            int(r.integers(2, max(3, W - 60))), int(r.integers(2, max(3, H - 8))))
            if t < self.g_end:
                img = self.micro(img, 0.25)
                s, mx, my = self.msg
                m = text_mask(s, ts)
                blit(img, dilate(np.pad(m, 1), 1), mx - 1, my - 1, (0, 0, 0))
                blit(img, m, mx, my, (255, 255, 255))
        else:
            bt = ph - 5.2
            if bt < 0.9:
                img = self.micro(img, bt / 0.9)
                if r.random() < bt:
                    y0 = int(r.integers(0, H - 4))
                    img[y0:y0 + H // 6] = 255 - img[y0:y0 + H // 6]
            elif bt < 1.6:
                img = self.micro(img, 0.6) * 0.22
                nch = W // 4
                for i in range(H // 6 + 1):
                    addr = ((int(t * 40) + i) * 16) & 0xFFFFFF
                    by = r.integers(0, 256, 8)
                    asc = "".join(r.choice(list("OMARCHY.#*/")) for _ in range(8))
                    line = (f"{addr:06X} " + " ".join(f"{x:02X}" for x in by) + " " + asc)[:nch]
                    y = i * 6
                    draw_text(img, line[:7], 0, y, (255, 40, 160), 1)
                    draw_text(img, line[7:], 28, y, (80, 255, 150) if i % 5 else (255, 255, 255), 1)
            elif bt < 2.3:
                bw = W / 7
                top = int(H * 0.66)
                for i, c in enumerate(BARS):
                    img[:top, int(i * bw):int((i + 1) * bw)] = c
                img[top:top + H // 12] = 0
                for i, c in enumerate(reversed(BARS)):
                    img[top:top + H // 12, int(i * bw):int((i + 1) * bw)] = np.array(c) * (i % 2)
                img[top + H // 12:] = (20, 20, 20)
                img[top + H // 12:, :int(W * 0.18)] = (0, 30, 80)
                img[top + H // 12:, int(W * 0.18):int(W * 0.36)] = (240, 240, 240)
                snow = r.random(H) < 0.12
                img[snow] = r.integers(0, 255, (int(snow.sum()), W, 1))
                s2 = ts * 2 if W > 120 else ts
                m = text_mask("NO SIGNAL", s2)
                bx, by = (W - m.shape[1]) // 2, (H - m.shape[0]) // 2
                img[by - 3 * s2:by + m.shape[0] + 3 * s2, bx - 4 * s2:bx + m.shape[1] + 4 * s2] = 0
                if int(t * 4) % 2 == 0:
                    blit(img, m, bx, by, (255, 255, 255))
            else:
                k = min(1.0, (bt - 2.3) / 0.6)
                off = int(H * (1 - k) ** 2)
                img = np.roll(img, off, 0)
                if 0 < off < H:
                    img[max(off - 2, 0):off] = 0
                img *= 1 + 0.8 * (1 - k)

        band = ((t * 0.12) % 1.4) * H - 0.2 * H
        img[(self.yy >= band) & (self.yy < band + max(2, H // 20))] += 16
        img[1::2] *= 0.84
        vg = 1 - 0.45 * (((self.xx - W / 2) / W) ** 2 + ((self.yy - H / 2) / H) ** 2)
        img *= vg[..., None]
        self.prev = np.clip(img, 0, 255)
        return img


# --------------------------------------------------------------------------- 7. HYPERSPACE


class Warp(Scene):
    name, title = "warp", "HYPERSPACE"
    CYCLE = 12.0

    def init(self):
        W, H, r = self.W, self.H, self.rng
        self.place(maxw=0.8, maxh=0.5)
        ys, xs = np.nonzero(self.L)
        self.home = np.stack([xs, ys], 1).astype(float)
        n = 280
        self.sx, self.sy, self.sz = r.uniform(-1, 1, n), r.uniform(-1, 1, n), r.uniform(0.05, 1, n)
        self.pal = grad([(0, (90, 0, 255)), (0.3, (0, 170, 255)), (0.5, (255, 255, 255)),
                         (0.7, (0, 255, 200)), (1, (90, 0, 255))])
        self.cyc = -1
        self.dist = np.hypot(self.xx - W / 2, self.yy - H / 2)
        self.bay = tile(bayer8(), H, W)

    def reset(self):
        r, W, H = self.rng, self.W, self.H
        n = len(self.home)
        ang = r.uniform(0, 2 * math.pi, n)
        rad = r.uniform(0.7, 1.3, n) * max(W, H)
        c = np.array([W / 2, H / 2])
        self.start = c + np.stack([np.cos(ang), np.sin(ang)], 1) * rad[:, None]
        self.delay = 0.6 + (self.home[:, 0] - self.lx) / max(self.lw, 1) * 1.2 + r.uniform(0, 0.3, n)
        self.pos = self.start.copy()
        self.vel = np.zeros_like(self.pos)
        self.boom = False

    def draw(self, dt):
        W, H, r, t = self.W, self.H, self.rng, self.t
        cyc, ph = int(t // self.CYCLE), t % self.CYCLE
        if cyc != self.cyc:
            self.cyc = cyc
            self.reset()
        cx, cy = W / 2, H / 2
        speed = (0.2 + 3.6 * (1 - float(smoothstep(ph, 0, 1.8)))
                 + 0.6 * float(smoothstep(ph, 8, 9) - smoothstep(ph, 9, 10))
                 + 3.6 * float(smoothstep(ph, 9.8, 11.2)))
        img = np.zeros((H, W, 3)) + np.array([2, 0, 10])

        # tunnel rings at lightspeed
        tk = max(1 - float(smoothstep(ph, 0, 1.4)), float(smoothstep(ph, 9.6, 10.6)))
        if tk > 0.01:
            for j in range(7):
                f = ((t * 1.6 + j / 7) % 1)
                rad = f ** 2 * max(W, H) * 0.9
                inten = np.clip(1 - np.abs(self.dist - rad) / (0.8 + rad * 0.05), 0, 1) * tk * (0.3 + 0.7 * f)
                m = inten > self.bay
                img[m] = np.maximum(img[m], hsv2rgb(0.62 + 0.15 * math.sin(j + t), 0.8, 0.55))

        # stars
        oz = self.sz.copy()
        self.sz -= speed * dt * 0.6
        dead = self.sz < 0.03
        nd = int(dead.sum())
        if nd:
            self.sx[dead], self.sy[dead], self.sz[dead] = r.uniform(-1, 1, nd), r.uniform(-1, 1, nd), 1.0
            oz[dead] = 1.0
        f = W * 0.32
        tail = np.minimum(self.sz + speed * 0.07, 1.4)
        warm = min(1.0, (speed - 0.2) / 3.0)
        for k in np.linspace(0, 1, 10):
            z = self.sz + (tail - self.sz) * k
            px = (cx + self.sx / z * f).astype(int)
            py = (cy + self.sy / z * f).astype(int)
            ok = (px >= 0) & (px < W) & (py >= 0) & (py < H)
            b = np.clip(1.15 - self.sz, 0, 1)[ok, None] * (1 - k * 0.85)
            head = np.array([200, 220, 255]) * (1 - warm) + np.array([180, 255, 255]) * warm
            tl = np.array([120, 140, 255]) * (1 - warm) + np.array([255, 60, 220]) * warm
            colr = head * (1 - k) + tl * k
            img[py[ok], px[ok]] = np.maximum(img[py[ok], px[ok]], colr * b)

        # logo particles
        hx = self.home[:, 0] * 0.012 + self.home[:, 1] * 0.006
        colv = self.pal[((hx - t * 0.25) % 1 * 255).astype(int)].copy()
        bright = 1.0
        if ph < 9.0:
            k = np.clip((ph - self.delay) / 1.5, 0, 1)
            e = 1 - (1 - k) ** 3
            d = self.home - self.start
            perp = np.stack([-d[:, 1], d[:, 0]], 1)
            self.pos = self.start + d * e[:, None] + perp * (np.sin(k * math.pi) * 0.35)[:, None]
            if 3.4 < ph < 8.0:  # breathing ripple + shimmer sweep + sparkle
                self.pos[:, 1] += np.round(np.sin(t * 3 - self.home[:, 0] * 0.12 / self.s) * 0.75 * self.s)
                sweep = self.lx + ((ph - 3.4) / 2.3 % 1) * (self.lw + 30) - 15
                near = np.abs(self.home[:, 0] - (self.home[:, 1] - self.ly) * 0.5 - sweep) < 2.5 * self.s
                colv[near] = colv[near] * 0.2 + 255 * 0.8
                sp = r.random(len(colv)) < 0.015
                colv[sp] = 255
                hold = float(smoothstep(ph, 3.4, 4.2) * (1 - smoothstep(ph, 7.4, 8.0)))
                rel = self.pos - np.array([cx, cy])
                sx = 1 + 0.06 * math.sin(t * 1.3) * hold
                self.pos[:, 0] = cx + rel[:, 0] * sx + math.sin(t * 0.9) * rel[:, 1] * 0.35 * hold
                self.pos[:, 1] = cy + rel[:, 1] * (1 + 0.05 * math.cos(t * 1.1) * hold)
            if ph > 8.0:
                q = (ph - 8.0)
                self.pos += r.normal(0, q * 0.9 * self.s, self.pos.shape)
                colv = colv + (255 - colv) * q
        else:
            if not self.boom:
                self.boom = True
                dv = self.home - np.array([cx, cy])
                nrm = np.linalg.norm(dv, axis=1, keepdims=True) + 1e-6
                self.vel = dv / nrm * r.uniform(25, 130, (len(dv), 1)) * (W / 120) + r.normal(0, 10, dv.shape)
            self.vel *= math.exp(-dt * 0.9)
            self.pos += self.vel * dt
            bright = max(0.0, 1 - (ph - 9.0) / 1.6)
            colv = colv * (1 - min(1, (ph - 9) * 2)) + np.array([255, 120, 220]) * min(1, (ph - 9) * 2)
        P = np.zeros((H, W, 3))
        trails = [(0.0, 1.0)]
        if ph >= 9.0:  # debris streaks, then sucked into the tunnel
            if ph > 9.6:
                pull = float(smoothstep(ph, 9.6, 10.6))
                d = np.array([cx, cy]) - self.pos
                self.vel = self.vel * (1 - pull * 0.15) + d * pull * 4 * dt * 30
            trails = [(k, 1 - k) for k in np.linspace(0, 1, 6)]
        for k, a in trails:
            pp = self.pos - self.vel * 0.06 * k
            ix, iy = np.round(pp[:, 0]).astype(int), np.round(pp[:, 1]).astype(int)
            ok = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
            P[iy[ok], ix[ok]] = np.maximum(P[iy[ok], ix[ok]], colv[ok] * bright * a)
        img = img + P + box_blur(P, 2) * 0.8 + box_blur(P, 5) * 0.45

        q = ph - 9.0
        if 0 <= q < 1.8:
            core = (np.exp(-self.dist / (W * 0.05))[..., None] * np.array([255, 255, 255])
                    + np.exp(-self.dist / (W * 0.22))[..., None] * np.array([70, 140, 255]))
            img += core * math.exp(-q * 4) * 1.4
            rr = q * max(W, H) * 0.7
            th = 1.2 + q * 3
            fade = max(0, 1 - q / 1.8)
            for off, col in ((-3 * self.s, (255, 40, 170)), (0, (255, 255, 255)), (3 * self.s, (40, 230, 255))):
                inten = np.clip(1 - np.abs(self.dist - (rr + off)) / th, 0, 1) * fade
                m = inten > self.bay
                img[m] = np.maximum(img[m], col)
            if q < 1.1:
                a = np.exp(-np.abs(self.xx[0] - cx) / (W * 0.3)) * math.exp(-q * 3)
                row = int(cy)
                img[row] += a[:, None] * np.array([200, 230, 255]) * 1.3
                img[row - 1] += a[:, None] * np.array([80, 120, 255]) * 0.5
                img[row + 1] += a[:, None] * np.array([80, 120, 255]) * 0.5
        return img


def dith(v, n, bay, w=0.4):
    """Quantise v (0..1) to n levels; only the middle `w` of each step is dithered, the rest is flat."""
    return np.clip(np.floor(v * (n - 1) + 0.5 + (bay - 0.5) * w), 0, n - 1).astype(int)


def qd(v, n, bay, w=0.4):
    return np.clip(np.floor(v * n + 0.5 + (bay - 0.5) * w), 0, n) / n


_DISC = {}


def disc(img, cx, cy, r, col):
    """Filled disc (cached mask per radius), clipped to the image."""
    H, W = img.shape[:2]
    key = int(round(r * 4))
    m = _DISC.get(key)
    if m is None:
        rr = max(key / 4, 0.5)
        R = int(math.ceil(rr))
        yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
        m = _DISC[key] = xx * xx + yy * yy <= rr * rr + 0.3
    R = m.shape[0] // 2
    x0, y0 = int(round(cx)) - R, int(round(cy)) - R
    xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + 2 * R + 1, W), min(y0 + 2 * R + 1, H)
    if xa < xb and ya < yb:
        img[ya:yb, xa:xb][m[ya - y0:yb - y0, xa - x0:xb - x0]] = col


def letter_ids(L):
    """Label logo columns by letter (1..n), 0 for the gaps between letters."""
    cols = L.any(0)
    ids = np.zeros(L.shape[1], int)
    k, prev = 0, False
    for x, c in enumerate(cols):
        if c and not prev:
            k += 1
        ids[x] = k if c else 0
        prev = bool(c)
    return ids, k


def crt(scene, img, scan=0.86, vig=0.5):
    if not hasattr(scene, "_vg"):
        W, H = scene.W, scene.H
        scene._vg = (1 - vig * (((scene.xx - W / 2) / W) ** 2 + ((scene.yy - H / 2) / H) ** 2))[..., None]
    img[1::2] *= scan
    return img * scene._vg


# --------------------------------------------------------------------------- 15. MOLTEN (lava lamp)


class Lava(Scene):
    name, title = "lava", "MOLTEN"

    def init(self):
        W, H, r = self.W, self.H, self.rng
        self.place(maxw=0.8, maxh=0.5)
        self.cs = cs = max(1, self.s)
        L = self.L
        self.bay = tile(bayer8(), H, W)
        n = 8
        self.bx = self.lx + (np.arange(n) + 0.5) / n * self.lw + r.uniform(-3, 3, n) * cs
        self.br = r.uniform(5.0, 8.0, n) * cs
        self.bp1, self.bp2 = r.uniform(0, 6.28, n), r.uniform(0, 6.28, n)
        self.bsp = r.uniform(0.16, 0.32, n)
        d = dilate(L, 1)
        self.out = d & ~L
        self.shadow = shift(d, 2 * cs, 2 * cs) & ~d
        self.top = L & ~shift(L, 1, 0)
        self.pal = np.array([(122, 16, 44), (176, 28, 44), (228, 62, 36), (255, 152, 40), (255, 236, 150)], float)
        self.bub = Particles()
        self.vg = 1 - 0.6 * (((self.xx - W / 2) / W) ** 2 + ((self.yy - H / 2) / H) ** 2)
        self.jit = (self.bay - 0.5) * (0.07 if cs > 1 else 0.0)

    def draw(self, dt):
        W, H, r, t, cs = self.W, self.H, self.rng, self.t, self.cs
        L = self.L
        f = np.zeros((H, W))
        for i in range(len(self.bx)):
            x = self.bx[i] + math.sin(t * 0.21 + self.bp1[i]) * 5 * cs
            y = self.lcy + math.sin(t * self.bsp[i] + self.bp2[i]) * self.lh * 0.7
            rr = self.br[i] * (1 + 0.1 * math.sin(t * 0.7 + self.bp1[i] * 3))
            f += rr * rr / ((self.xx - x) ** 2 + (self.yy - y) ** 2 + 2.0) * 0.62
        # every 20s one big slow blob of heat crosses the whole word
        sp = (t % 20.0) / 7.0
        if sp < 1.25:
            sx = self.lx - 12 * cs + sp * (self.lw + 24 * cs)
            rr = 11.0 * cs
            f += rr * rr / ((self.xx - sx) ** 2 + ((self.yy - self.lcy) * 0.8) ** 2 + 2.0) * 0.8
        blob = f.copy()
        f += np.clip((self.yy - (self.ly + self.lh - 3.5 * cs)) / (3.0 * cs), 0, 1.2) * 0.9
        f += np.clip(((self.ly + 1.5 * cs) - self.yy) / (3.0 * cs), 0, 1) * 0.35
        fj = f + self.jit
        idx = (fj > 0.42).astype(int) + (fj > 0.68) + (fj > 1.05) + (fj > 1.7)
        fill = self.pal[idx]

        img = np.zeros((H, W, 3)) + np.array([8, 3, 10])
        ghost = qd(np.clip(blob / 2.2, 0, 1) * self.vg, 4, self.bay, 0.6)
        img += ghost[..., None] * np.array([34, 6, 16])
        lum = (idx / 4.0) * L
        glow = np.clip(box_blur(box_blur(lum, 4 * cs), 4 * cs) * 2.2 + box_blur(box_blur(lum, 10 * cs), 10 * cs) * 1.6, 0, 1)
        img += qd(glow, 6, self.bay, 0.5)[..., None] * np.array([255, 84, 24]) * 0.62
        img[self.shadow] *= 0.25
        img[self.out] = (4, 1, 6)
        img[L] = fill[L]
        img[self.top] = np.minimum(img[self.top] + np.array([44, 30, 24]), 255)

        if r.random() < dt * 2:
            ys, xs = np.nonzero(L & (self.yy > self.ly + self.lh * 0.6))
            if len(ys):
                k = int(r.integers(0, len(ys)))
                self.bub.add([[xs[k], ys[k]]], [[0, -r.uniform(3, 7) * cs]], (255, 244, 200), r.uniform(1.2, 2.6))
        self.bub.update(dt)
        if len(self.bub.l):
            bx = np.clip(np.round(self.bub.p[:, 0]).astype(int), 0, W - 1)
            by = np.clip(np.round(self.bub.p[:, 1]).astype(int), 0, H - 1)
            ok = L[by, bx]
            img[by[ok], bx[ok]] = np.minimum(img[by[ok], bx[ok]] + 60, 255)
        return crt(self, img, 0.88, 0.4)


# --------------------------------------------------------------------------- 19. BIOLUME (glowing tide)


class Tide(Scene):
    name, title = "tide", "BIOLUME"
    CYCLE = 12.0

    def init(self):
        W, H, r = self.W, self.H, self.rng
        self.place(maxw=0.8, maxh=0.5)
        self.cs = cs = max(1, self.s)
        L = self.L
        self.bay = tile(bayer8(), H, W)
        d = dilate(L, 1)
        self.out = d & ~L
        self.shadow = shift(d, 2 * cs, 2 * cs) & ~d
        self.top = L & ~shift(L, 1, 0)
        n = W * H // 40
        self.px, self.py = r.integers(0, W, n), r.integers(0, H, n)
        self.pph = r.uniform(0, 6.28, n)
        self.wake = np.zeros((H, W))
        self.pal = np.array([(0, 84, 96), (0, 128, 132), (0, 182, 160), (70, 238, 204), (220, 255, 242)], float)
        self.xr, self.yr = self.xx * 160 / W, self.yy * 90 / H
        near = np.clip(box_blur(box_blur(L.astype(float), 6 * cs), 6 * cs) * 2.5, 0, 1)
        self.near = near

    def swell(self, fx, sign, strength):
        """One wave front at x = fx (per row), travelling in direction `sign`; returns (crest, behind-glow)."""
        dx = (self.xx - fx) * sign                                   # > 0 ahead of the front
        ahead = np.exp(-np.maximum(dx, 0) / (3.0 * self.cs))
        behind = np.exp(np.minimum(dx, 0) / (28.0 * self.cs))
        return (np.abs(dx) < 0.8 * self.cs) * strength, np.where(dx > 0, ahead, behind) * strength

    def draw(self, dt):
        W, H, r, t, cs = self.W, self.H, self.rng, self.t, self.cs
        L = self.L
        xr, yr = self.xr, self.yr
        w = np.sin(xr * 0.07 - t * 0.8 + np.sin(yr * 0.09 + t * 0.25) * 1.6) + 0.7 * np.sin(xr * 0.031 + yr * 0.11 + t * 0.45) \
            + 0.4 * np.sin(yr * 0.21 - t * 0.6 + xr * 0.02)
        v = (w + 2.1) / 4.2
        ph = t % self.CYCLE
        self.wake *= math.exp(-dt * 0.4)
        crest = np.zeros((H, W))
        wob = np.sin(yr * 0.12 + t * 0.7) * 5 * cs + np.sin(yr * 0.05) * 9 * cs
        if ph < 5.5:                                                  # the big swell, left to right
            c, g = self.swell(-0.15 * W + ph / 5.0 * 1.3 * W + wob, 1, 1.0)
            crest, self.wake = np.maximum(crest, c), np.maximum(self.wake, g)
        if 6.5 < ph < 11.5:                                           # a weaker return swell, right to left
            c, g = self.swell(1.15 * W - (ph - 6.5) / 5.0 * 1.3 * W - wob, -1, 0.55)
            crest, self.wake = np.maximum(crest, c), np.maximum(self.wake, g)

        img = np.zeros((H, W, 3)) + np.array([1, 6, 12])
        img += qd(v ** 2, 3, self.bay, 0.6)[..., None] * np.array([0, 12, 18])
        img += qd(self.wake * 0.8, 4, self.bay, 0.7)[..., None] * np.array([0, 56, 52])
        # plankton flare under crests, in the wake, and where the letters disturb the water
        stir = self.near[self.py, self.px] * (0.35 + 0.25 * np.sin(t * 1.7 + self.pph))
        pv = v[self.py, self.px]
        pb = np.clip((pv - 0.66) * 4, 0, 1) * (0.6 + 0.4 * np.sin(t * 3 + self.pph)) + self.wake[self.py, self.px] * 1.1 + stir
        pb = np.clip(pb, 0, 1)
        col = np.array([10, 120, 110]) * (1 - pb[:, None]) + np.array([180, 255, 232]) * pb[:, None]
        on = pb > 0.08
        img[self.py[on], self.px[on]] = np.maximum(img[self.py[on], self.px[on]], col[on] * pb[on, None] ** 0.5)
        behind = (self.wake > 0.4) & (self.bay < (self.wake - 0.4) * 1.4)
        img[behind] = np.maximum(img[behind], (60, 200, 180))
        cr = crest > 0.3
        img[cr] = np.maximum(img, np.array([120, 236, 214]) + np.array([110, 19, 32]) * np.clip(crest * 2 - 1, 0, 1)[..., None])[cr]

        f = v * 0.8 + 0.2 + self.wake * 0.75
        idx = (f > 0.42).astype(int) + (f > 0.58) + (f > 0.76) + (f > 1.05)
        fill = self.pal[idx]
        lum = (0.4 + idx / 4.0 * 0.6) * L
        glow = np.clip(box_blur(box_blur(lum, 4 * cs), 4 * cs) * 2.0 + box_blur(box_blur(lum, 10 * cs), 10 * cs) * 1.4, 0, 1)
        img += qd(glow, 6, self.bay, 0.5)[..., None] * np.array([0, 170, 150]) * 0.6
        img[self.shadow] *= 0.3
        img[self.out] = (0, 4, 8)
        img[L] = fill[L]
        img[self.top] = np.minimum(img[self.top] + np.array([40, 50, 44]), 255)
        return crt(self, img, 0.88, 0.45)


def hblur(a, r):
    """Box blur along x only (streaks, fibres)."""
    if r <= 0:
        return a.astype(float)
    k = 2 * r + 1
    c = np.cumsum(np.pad(a.astype(float), ((0, 0), (r + 1, r)), mode="edge"), axis=1)
    return (c[:, k:] - c[:, :-k]) / k


class Quiet(Scene):
    """Shared set-up: logo big and centred, flat background, letter ids and edges."""
    BG = (0, 0, 0)

    def init(self):
        W, H = self.W, self.H
        self.place(maxw=0.84, maxh=0.46)
        self.cs = max(1, self.s)
        self.bay = tile(bayer8(), H, W)
        L = self.L
        self.ids, self.nl = letter_ids(L)
        self.lid = np.where(L, self.ids[None, :], 0)
        self.top = L & ~shift(L, 1, 0)
        self.bot = L & ~shift(L, -1, 0)
        self.out = dilate(L, 1) & ~L
        self.rel = np.clip((self.yy - self.ly) / max(self.lh - 1, 1), 0, 1)
        self.bgimg = np.zeros((H, W, 3)) + np.array(self.BG, float)
        self.setup()

    def setup(self):
        pass

    def halo(self, lum, col, r1=3, r2=8, k=1.0):
        cs = self.cs
        g = box_blur(lum, r1 * cs)
        g = box_blur(g, r1 * cs) * 1.5 + box_blur(g, r2 * cs) * 0.8
        return qd(np.clip(g * k, 0, 1), 6, self.bay, 0.5)[..., None] * np.array(col)


# --------------------------------------------------------------------------- 28. SUMI-E


SEAL = ["##.##", "#...#", ".###.", "#...#", "##.##"]


class Ink(Quiet):
    name, title = "ink", "SUMI-E"
    BG = (10, 10, 14)
    CYCLE = 24.0

    def setup(self):
        W, H, r, cs, L = self.W, self.H, self.rng, self.cs, self.L
        self.bgimg *= (1 - 0.3 * (((self.xx - W / 2) / W) ** 2 * 2 + ((self.yy - H / 2) / H) ** 2 * 2))[..., None]
        x0 = np.zeros(self.nl + 1)
        x1 = np.ones(self.nl + 1)
        for k in range(1, self.nl + 1):
            c = np.nonzero(self.ids == k)[0]
            x0[k], x1[k] = c[0], c[-1] + 1
        self.x0, self.x1 = x0, x1
        frac = np.clip((self.xx - x0[self.lid]) / (x1 - x0)[self.lid], 0, 1)
        streak = hblur(r.random((H, W)), 3 * cs)
        streak = (streak - streak.mean()) / (streak.std() + 1e-6)
        self.dry = L & (streak > 2.2 - frac ** 1.5 * 1.6)
        self.bleed = dilate(L, max(1, cs)) & ~L & (box_blur(r.random((H, W)), 1) < 0.5)
        w = box_blur(r.random((H, W)), 2 * cs + 1)
        self.wash = (w - w.min()) / (np.ptp(w) + 1e-6)
        self.wob = (np.sin(np.arange(H) * 0.7) + np.sin(np.arange(H) * 0.29) * 1.5) * cs
        S = 7 * max(1, cs // 2 + 1)
        self.seal = (min(self.lx + self.lw - S, W - S - 2), min(self.ly + self.lh + 3 * cs, H - S - 2), S)

    def draw(self, dt):
        W, H, t, cs, L = self.W, self.H, self.t, self.cs, self.L
        ph = t % self.CYCLE
        k = np.arange(self.nl + 1)
        st = 1.0 + (k - 1) * 0.7
        fx = (self.x0 + (ph - st) / 0.55 * (self.x1 - self.x0 + 2 * cs))[self.lid] + self.wob[:, None]
        painted = L & (self.xx < fx)
        wet = painted & (self.xx > fx - 3 * cs)
        alive = qd(np.clip((self.wash - smoothstep(ph, 18, 22.5)) * 5, 0, 1), 3, self.bay, 0.6)
        img = self.bgimg.copy()
        bl = self.bleed & dilate(painted, max(1, cs)) & (alive > 0)
        img[bl] = img[bl] * 0.55 + np.array([70, 68, 66]) * 0.45
        ink = np.zeros((H, W, 3)) + np.array([222, 214, 198])
        ink[self.dry] = (110, 106, 100)
        ink[wet] = (255, 252, 244)
        a = (alive * painted)[..., None]
        img = img * (1 - a) + ink * a
        sx, sy, S = self.seal
        ts = st[-1] + 1.4
        press = float(smoothstep(ph, ts, ts + 0.3))
        if press > 0 and sx > 0 and sy > 0:
            sq = (self.bay[sy:sy + S, sx:sx + S] < press * 1.15) & (alive[sy:sy + S, sx:sx + S] > 0)
            img[sy:sy + S, sx:sx + S][sq] = (186, 40, 36)
            g = np.zeros((S, S), bool)
            g[1, 1:-1] = g[-2, 1:-1] = g[1:-1, 1] = g[1:-1, -2] = True
            gm = np.kron(np.array([[c == "#" for c in row] for row in SEAL]), np.ones((max(1, (S - 4) // 5),) * 2, bool))
            o = (S - gm.shape[0]) // 2
            g[o:o + gm.shape[0], o:o + gm.shape[1]] |= gm
            img[sy:sy + S, sx:sx + S][g & sq] = (236, 210, 196)
        return img


# --------------------------------------------------------------------------- 29. EMBERS

# glowing coal: deep red -> orange -> amber -> white-hot (only the cracks at the peak reach the top)
EMBERPAL = grad([(0, (70, 10, 6)), (0.3, (150, 26, 8)), (0.55, (226, 78, 16)), (0.78, (255, 150, 40)),
                 (0.92, (255, 214, 120)), (1, (255, 246, 214))])
EMB_ASH = np.array([[38, 28, 26], [52, 41, 38], [68, 56, 51], [96, 82, 74]], float)


def emb_heat(ph):
    """The breath of the fire over one cycle: cool ash -> rekindle -> glow -> settle back."""
    up = smoothstep(ph, 4.0, 12.0)
    down = smoothstep(ph, 20.0, 28.0)
    return 0.08 + 0.92 * up * (1 - down)


class Embers(Quiet):
    name, title = "embers", "EMBERS"
    BG = (4, 2, 2)
    CYCLE = 32.0

    def setup(self):
        H, W, r, cs, L = self.H, self.W, self.rng, self.cs, self.L
        ys, xs = np.nonzero(L)
        n = len(ys)
        lid = self.lid[ys, xs]
        dmin = max(3.0, 0.42 * self.lh)                           # few, big lumps: the letters stay whole
        d1, d2, own = np.full(n, 99.0), np.full(n, 99.0), np.zeros(n, int)
        sx, sy = [], []
        for li in range(1, self.nl + 1):                          # each letter cracks on its own
            idx = np.nonzero(lid == li)[0]
            if not len(idx):
                continue
            seeds = []
            for j in r.permutation(idx):
                if all((xs[j] - xs[q]) ** 2 + ((ys[j] - ys[q]) * 1.2) ** 2 >= dmin * dmin for q in seeds):
                    seeds.append(j)
                    if len(seeds) >= 6:
                        break
            base = len(sx)
            sx += [float(xs[q]) for q in seeds]
            sy += [float(ys[q]) for q in seeds]
            d = np.hypot(xs[idx, None] - xs[seeds][None], (ys[idx, None] - ys[seeds][None]) * 1.2)
            if len(seeds) > 1:
                p = np.partition(d, 1, axis=1)
                d1[idx], d2[idx] = p[:, 0], p[:, 1]
            own[idx] = base + np.argmin(d, 1)
        sx, sy = np.array(sx), np.array(sy)
        m = len(sx)
        gap = d2 - d1
        self.crack = np.zeros((H, W), bool)
        self.crack[ys, xs] = gap < 1.25 + 0.3 * (cs - 1)
        # depth into the lump: 0 at the crack, 1 at the lump's heart
        mx = np.zeros(m)
        np.maximum.at(mx, own, gap)
        u = np.zeros((H, W))
        u[ys, xs] = np.clip(gap / np.maximum(mx[own], 1e-3), 0, 1)
        self.u = u
        self.own = np.zeros((H, W), int)
        self.own[ys, xs] = own
        self.ys, self.xs = ys, xs
        # each lump has its own temper, and the breath reaches it a little later the further right it sits
        self.hot = r.uniform(0.7, 1.0, m)
        self.lag = (sx - self.lx) / max(self.lw, 1) * 3.0 + r.uniform(0, 1.2, m)
        self.ph0 = r.uniform(0, 6.28, m)
        self.sparks = Particles()
        self.tops = np.nonzero(self.top)
        # flat per-pixel views of the coal, so a frame only touches letter pixels
        self.po = own
        self.cx = sx
        self.pu = self.u[ys, xs]
        self.pb = self.bay[ys, xs]
        self.pcr = self.crack[ys, xs]
        self.ptop = self.top[ys, xs]
        self.pbot = self.bot[ys, xs]

    def draw(self, dt):
        t, cs, r, L = self.t, self.cs, self.rng, self.L
        ph = (t + 2.0) % self.CYCLE
        G = emb_heat(ph - self.lag)                                # per lump
        breath = 0.88 + 0.12 * np.sin(t * 0.55 + self.ph0)
        T = G * self.hot * breath
        # during the hold a slow gust rolls across the bed and the lumps flare as it passes
        gx = self.lx - 0.3 * self.lw + ((ph - 13.0) / 7.0) * 1.6 * self.lw
        T = T * (1 + 0.28 * np.exp(-((self.cx - gx) / (0.22 * self.lw)) ** 2) * (G > 0.6))
        front = 0.14 + 0.96 * smoothstep(G, 0.1, 0.85)            # how far the glow has eaten into the ash
        ys, xs, o, u, bay, cr = self.ys, self.xs, self.po, self.pu, self.pb, self.pcr
        fo = front[o]
        edge = u - fo + (bay - 0.5) * 0.05
        glow = edge < 0
        # heat inside a lump: hottest along its cracks, cooler toward the heart
        hv = (0.18 + 0.82 * T[o]) * (1 - (0.62 - 0.3 * T[o]) * np.clip(u / np.maximum(fo, 0.2), 0, 1)) * (0.35 + 0.65 * np.minimum(T[o] * 3, 1))
        ct = 0.24 + 0.1 * np.sin(t * 0.45 + self.ph0[o] * 2) * (1 - G[o]) + 0.7 * G[o] * (0.88 + 0.12 * np.sin(t * 0.9 + self.ph0[o]))
        heat = np.where(glow, np.minimum(hv, 0.78), 0.0)
        heat[cr] = np.maximum(heat[cr], ct[cr])
        g = float(emb_heat(ph - 1.5))
        img = self.bgimg.copy()
        hm = np.zeros(L.shape)
        hm[ys, xs] = heat
        img += self.halo(hm * (0.25 + 0.75 * g), (150, 44, 10), 3, 8, 0.65)
        img[self.out] *= 0.35
        # ash skin: charcoal, a paler crust toward each lump's heart, a pale lip on top and along the creeping edge
        ai = dith(0.2 + 0.55 * u, 3, bay, 0.25)
        ai = np.where(self.ptop, 3, ai)
        ai = np.where((edge >= 0) & (edge < 0.09), 3, ai)
        col = EMB_ASH[ai]
        q = qd(np.clip(heat, 0, 1), 5, bay, 0.3)
        hot = (glow | cr) & (q > 0)
        col[hot] = EMBERPAL[(q[hot] * 255).astype(int)]
        col[self.pbot & ~hot] *= 0.7
        img[ys, xs] = col
        # sparks lift off the hottest tops
        if len(self.tops[0]) and r.random() < dt * (0.2 + 3.5 * g * g) * self.W / 160:
            i = r.integers(len(self.tops[0]))
            self.sparks.add([[self.tops[1][i], self.tops[0][i] - 1]],
                            [[r.uniform(-1.5, 1.5), -r.uniform(4, 9) * self.H / 90]],
                            (255, 190, 90) if r.random() < 0.6 else (255, 130, 40), r.uniform(1.8, 3.4))
        self.sparks.v[:, 0] += np.sin(t * 1.3 + self.sparks.p[:, 1] * 0.2) * dt * 3
        self.sparks.update(dt, drag=0.25)
        self.sparks.draw(img)
        return img


WAT_HI, WAT_LO = (226, 250, 255), (96, 184, 222)


# --------------------------------------------------------------------------- 31. DEEP FIELD

# Each cycle a new star ignites somewhere inside the word and lights up the nebula around it.
# Palettes are (warm ramp, cool ramp, star tint, halo tint); ramps run dark -> bright.
COS_PALETTES = [
    ([(0, (14, 10, 40)), (0.25, (60, 18, 96)), (0.5, (170, 36, 130)), (0.72, (255, 104, 120)), (0.88, (255, 190, 160)), (1, (255, 244, 226))],
     [(0, (8, 14, 44)), (0.25, (18, 48, 120)), (0.5, (30, 120, 190)), (0.72, (70, 200, 230)), (0.88, (170, 240, 250)), (1, (240, 255, 255))],
     (255, 236, 220), (150, 40, 120)),
    ([(0, (22, 14, 36)), (0.25, (92, 46, 34)), (0.5, (170, 96, 30)), (0.72, (240, 170, 60)), (0.88, (255, 222, 140)), (1, (255, 250, 220))],
     [(0, (8, 24, 44)), (0.25, (14, 72, 90)), (0.5, (20, 130, 136)), (0.72, (70, 200, 186)), (0.88, (170, 245, 225)), (1, (240, 255, 248))],
     (255, 246, 214), (120, 80, 30)),
    ([(0, (10, 12, 44)), (0.25, (30, 36, 130)), (0.5, (70, 90, 220)), (0.72, (130, 160, 255)), (0.88, (200, 214, 255)), (1, (250, 252, 255))],
     [(0, (18, 10, 44)), (0.25, (64, 30, 130)), (0.5, (130, 70, 210)), (0.72, (200, 140, 255)), (0.88, (236, 206, 255)), (1, (255, 248, 255))],
     (226, 236, 255), (60, 70, 170)),
]
COS_LEVELS = 9


def COS_wrap_blur(a, r):
    """Box blur that wraps horizontally (tileable noise)."""
    k = 2 * r + 1
    c = np.cumsum(np.pad(a, ((0, 0), (r + 1, r)), mode="wrap"), axis=1)
    a = (c[:, k:] - c[:, :-k]) / k
    c = np.cumsum(np.pad(a, ((r + 1, r), (0, 0)), mode="edge"), axis=0)
    return (c[k:] - c[:-k]) / k


def COS_fbm(rng, H, W, base):
    out = np.zeros((H, W))
    amp, tot = 1.0, 0.0
    for o in range(3):
        r = max(1, int(base / 2 ** o))
        n = rng.random((H, W))
        for _ in range(2):
            n = COS_wrap_blur(n, r)
        n = (n - n.mean()) / (n.std() + 1e-9)
        out += n * amp
        tot += amp
        amp *= 0.45
    out /= tot
    return np.clip(0.5 + out * 0.32, 0, 1)


class Cosmos(Quiet):
    name, title = "cosmos", "DEEP FIELD"
    BG = (0, 0, 0)
    CYCLE = 32.0

    def setup(self):
        W, H, r, L, cs = self.W, self.H, self.rng, self.L, self.cs
        self.ys, self.xs = np.nonzero(L)
        self.TW = TW = 2 * W
        base = max(4, self.lh * 0.5)
        self.nebA = COS_fbm(r, H, TW, base)
        self.nebB = COS_fbm(r, H, TW, base * 0.7)
        self.hueN = COS_fbm(r, H, TW, base * 1.4)
        d = COS_fbm(r, H, TW, base * 0.6)
        self.dust = smoothstep(1 - np.abs(2 * d - 1), 0.8, 0.97)       # thin ridged dark lanes
        self.ramps = [(grad(a), grad(b), np.array(s, float), np.array(h, float)) for a, b, s, h in COS_PALETTES]
        # stars at three depths, scattered over a band wider than the screen so they wrap smoothly
        n = max(120, int(L.sum()) // (5 * cs))
        self.sx, self.sy = r.uniform(0, W, n), r.integers(0, H, n)
        self.sl = r.choice(3, n, p=[0.6, 0.3, 0.1])
        self.sph = r.uniform(0, 6.28, n)
        self.scol = np.array([(255, 255, 255), (190, 214, 255), (255, 226, 190), (255, 200, 200)], float)[r.integers(0, 4, n)]
        self.shoot = None
        self.cyc = -1
        # the space that is open to the screen edge: the glow may spill there, never into the counters
        ext = np.zeros((H, W), bool)
        ext[0, :] = ext[-1, :] = ext[:, 0] = ext[:, -1] = True
        ext &= ~L
        while True:
            nx = dilate(ext, 4, diag=False) & ~L
            if (nx == ext).all():
                break
            ext = nx
        self.glowok = ext & ~self.out
        self.dw = 0.3 if cs == 1 else 0.4

    def new_cycle(self, k):
        r, ys, xs, cs = self.rng, self.ys, self.xs, self.cs
        # ignite in a thick part of a letter, away from the far ends
        thick = dilate(~self.L, max(1, cs)) & self.L
        cand = np.nonzero(self.L & ~thick & (self.xx > self.lx + self.lw * 0.15) & (self.xx < self.lx + self.lw * 0.85))
        if len(cand[0]) == 0:
            cand = (ys, xs)
        i = r.integers(len(cand[0]))
        self.star = (float(cand[1][i]), float(cand[0][i]))
        self.pal = self.ramps[k % len(self.ramps)]
        self.meteor_t = r.uniform(14, 19)
        self.shoot = None

    def draw(self, dt):
        W, H, t, cs, r, L = self.W, self.H, self.t, self.cs, self.rng, self.L
        ys, xs, TW = self.ys, self.xs, self.TW
        k = int(t // self.CYCLE)
        ph = t % self.CYCLE
        if k != self.cyc:
            self.cyc = k
            self.new_cycle(k)
        warm, cool, stint, htint = self.pal
        sx, sy = self.star
        # --- the arc: night -> ignition -> light front sweeps out -> hold -> fades back to night
        ign = float(smoothstep(ph, 2.5, 4.0))
        fade = float(smoothstep(ph, 23.0, 29.0))
        Rmax = math.hypot(self.lw, self.lh)
        R = Rmax * float(smoothstep(ph, 3.5, 12.0)) ** 0.8
        rr = np.hypot(xs - sx, (ys - sy) * 1.3)
        fw = 6.0 * cs
        lit = np.clip((R - rr) / fw, 0, 1) * (1 - fade)
        front = np.exp(-((rr - R) / (2.5 * cs)) ** 2) * (R < Rmax * 0.98) * ign * (1 - fade)
        near = np.exp(-rr / (0.35 * self.lw))
        # --- nebula: two layers drifting at different speeds = depth
        o1 = int(t * 0.8 * cs) % TW
        o2 = int(t * 1.6 * cs) % TW
        xa, xb = (xs + o1) % TW, (xs + o2) % TW
        dens = 0.62 * self.nebA[ys, xa] + 0.38 * self.nebB[ys, xb]
        dens = np.clip((dens - 0.28) / 0.5, 0, 1) * (1 - 0.85 * self.dust[ys, xb])
        I = 0.34 + lit * (0.22 + 0.5 * near)
        v = dens * I + front * 0.32 * (0.3 + dens) + ign * (1 - fade * 0.7) * np.exp(-(rr / (7.0 * cs)) ** 2) * 0.4
        v = np.clip(v, 0, 1)
        bay = self.bay[ys, xs]
        vi = (qd(v, COS_LEVELS - 1, bay, self.dw) * 255).astype(int)
        # hot ionised gas near the star, cooler reflection gas further out
        ie = ign * (1 - fade)
        hm = ie * rr / (0.45 * self.lw) + (1 - ie) * 0.75 + (self.hueN[ys, (xs + o1) % TW] - 0.5) * 1.6
        m = qd(np.clip(hm - 0.3, 0, 1), 1, bay, 0.55 if cs > 1 else 0.15)[:, None]
        col = warm[vi] * (1 - m) + cool[vi] * m
        # during the dark lull the next cycle's palette dissolves in, pixel by pixel
        wv = float(smoothstep(ph, 28.0, 31.8))
        if wv > 0:
            w2, c2, _, _ = self.ramps[(k + 1) % len(self.ramps)]
            sw = (bay < wv)[:, None]
            col = np.where(sw, w2[vi] * (1 - m) + c2[vi] * m, col)
        img = self.bgimg.copy()
        # soft glow spilling out of the letters
        lum = np.zeros((H, W))
        lum[ys, xs] = v * v
        img[self.glowok] += self.halo(lum, htint * (0.35 + 0.65 * (1 - fade) * ign), 2, 7, 0.5)[self.glowok]
        img[ys, xs] = col
        # --- stars at three depths
        x = (self.sx - t * (0.5 + self.sl * 1.2) * cs) % W
        xi = x.astype(int)
        on = L[self.sy, xi]
        tw = 0.55 + 0.45 * np.sin(t * (0.7 + 0.5 * self.sl) + self.sph)
        b = (0.4 + self.sl * 0.3) * tw
        py, px = self.sy[on], xi[on]
        img[py, px] = np.maximum(img[py, px] * 0.5 + self.scol[on] * b[on, None], img[py, px])
        big = on & (self.sl == 2) & (tw > 0.8)
        if cs > 1:
            for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                qy, qx = np.clip(self.sy[big] + dy, 0, H - 1), np.clip(xi[big] + dx, 0, W - 1)
                ok = L[qy, qx]
                img[qy[ok], qx[ok]] = np.maximum(img[qy[ok], qx[ok]], self.scol[big][ok] * 0.5)
        # --- the newborn star: core + diffraction spikes, clipped to the letters
        s = ign * (1 - fade * 0.8) * (0.9 + 0.1 * math.sin(t * 1.3))
        if s > 0.02:
            ix, iy = int(sx), int(sy)
            ln = int((4 + 6 * s) * cs)
            for j in range(-ln, ln + 1):
                a = s * (1 - abs(j) / (ln + 1)) ** 1.2
                a = math.floor(a * 4 + 0.5) / 4                      # stepped falloff: pixel-art spikes
                for qy, qx in ((iy, ix + j), (iy + j, ix)):
                    if 0 <= qy < H and 0 <= qx < W and L[qy, qx]:
                        img[qy, qx] = np.minimum(img[qy, qx] * (1 - a * 0.5) + stint * a, 255)
            if cs > 1:
                for dy, dx in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                    for j in range(1, cs + 1):
                        qy, qx = iy + dy * j, ix + dx * j
                        if 0 <= qy < H and 0 <= qx < W and L[qy, qx]:
                            img[qy, qx] = np.minimum(img[qy, qx] + stint * s * 0.35, 255)
            disc(img, sx, sy, max(0.5, (0.4 + 0.5 * s) * cs), stint * min(1, s * 1.2))
            if s > 0.5:
                img[iy, ix] = (255, 255, 255)
        # --- one meteor per cycle, during the hold
        if self.shoot is None and self.meteor_t < ph < self.meteor_t + 0.1:
            self.shoot = [r.uniform(self.lx + self.lw * 0.4, self.lx + self.lw), self.ly - 2.0 * cs, t]
        if self.shoot:
            x0, y0, t0 = self.shoot
            a = t - t0
            if a > 2.4:
                self.shoot = None
            else:
                # a continuous streak: bright head, a tail that cools from white through the star tint
                n = 24 * cs
                j = np.arange(n)
                ux, uy = -0.94, 0.34
                d = a * 48 * cs - j
                ok = d > 0
                px_ = (x0 + ux * d).astype(int)
                py_ = (y0 + uy * d).astype(int)
                ok &= (px_ >= 0) & (px_ < W) & (py_ >= 0) & (py_ < H)
                px_, py_, jj = px_[ok], py_[ok], j[ok]
                ins = L[py_, px_]
                f = np.floor((1 - jj / n) ** 1.4 * 4 + 0.5)[:, None] / 4
                c = np.array([255, 252, 240]) * f + stint * 0.3 * (1 - f)
                px_, py_, f, c = px_[ins], py_[ins], f[ins], c[ins]
                img[py_, px_] = np.maximum(img[py_, px_], c * np.minimum(1, f * 2.5))
        return img


# --------------------------------------------------------------------------- 32. DAYLIGHT


# One day through the letters. Each key: phase p (0..1), five sky colours top->bottom,
# cloud (lit, shade), cloud cover, star visibility, horizon glow strength.
DAY_KEYS = [
    (0.00, [(12, 14, 46), (18, 24, 64), (28, 36, 86), (40, 50, 110), (58, 66, 136)], (92, 98, 150), (40, 44, 86), 0.22, 1.0, 0.0),
    (0.08, [(8, 10, 34), (16, 18, 54), (28, 30, 78), (46, 42, 102), (70, 56, 122)], (110, 96, 150), (44, 40, 86), 0.24, 0.9, 0.1),
    (0.15, [(20, 22, 66), (52, 46, 116), (120, 72, 140), (214, 104, 120), (255, 168, 104)], (220, 130, 150), (70, 46, 104), 0.32, 0.35, 0.8),
    (0.21, [(36, 62, 150), (92, 104, 184), (196, 130, 156), (252, 170, 120), (255, 218, 140)], (255, 210, 190), (150, 104, 150), 0.36, 0.0, 1.0),
    (0.28, [(44, 104, 204), (76, 140, 226), (124, 176, 238), (196, 210, 228), (250, 228, 180)], (255, 250, 240), (168, 178, 214), 0.42, 0.0, 0.35),
    (0.36, [(30, 86, 204), (44, 112, 226), (70, 146, 240), (108, 180, 248), (156, 214, 252)], (255, 255, 255), (170, 192, 228), 0.48, 0.0, 0.0),
    (0.50, [(30, 86, 204), (44, 112, 226), (70, 146, 240), (108, 180, 248), (156, 214, 252)], (255, 255, 255), (170, 192, 228), 0.48, 0.0, 0.0),
    (0.58, [(40, 90, 196), (66, 122, 216), (112, 158, 228), (196, 196, 206), (252, 214, 156)], (255, 242, 214), (180, 170, 196), 0.44, 0.0, 0.4),
    (0.65, [(44, 34, 110), (112, 52, 140), (214, 76, 112), (252, 126, 70), (255, 196, 92)], (255, 196, 120), (140, 52, 98), 0.40, 0.0, 1.0),
    (0.71, [(18, 18, 60), (50, 32, 96), (110, 44, 112), (186, 66, 92), (236, 112, 70)], (236, 104, 96), (66, 28, 78), 0.34, 0.25, 0.7),
    (0.79, [(12, 14, 46), (18, 24, 64), (28, 36, 86), (40, 50, 110), (58, 66, 136)], (92, 98, 150), (40, 44, 86), 0.22, 1.0, 0.05),
    (1.00, [(12, 14, 46), (18, 24, 64), (28, 36, 86), (40, 50, 110), (58, 66, 136)], (92, 98, 150), (40, 44, 86), 0.22, 1.0, 0.0),
]
DAY_NB = 9          # sky bands across the letter height


def day_interp(p):
    for k0, k1 in zip(DAY_KEYS, DAY_KEYS[1:]):
        if k0[0] <= p <= k1[0]:
            a = (p - k0[0]) / (k1[0] - k0[0])
            a = a * a * (3 - 2 * a)
            mix = lambda u, v: np.array(u, float) * (1 - a) + np.array(v, float) * a
            return (mix(k0[1], k1[1]), mix(k0[2], k1[2]), mix(k0[3], k1[3]),
                    *(k0[i] * (1 - a) + k1[i] * a for i in (4, 5, 6)))
    k = DAY_KEYS[0]
    return np.array(k[1], float), np.array(k[2], float), np.array(k[3], float), k[4], k[5], k[6]


class Daylight(Quiet):
    name, title = "daylight", "DAYLIGHT"
    BG = (0, 0, 0)
    CYCLE = 34.0
    P0 = 0.03          # start just before dawn so a single viewing gets the whole arc

    def setup(self):
        W, H, r, cs, L = self.W, self.H, self.rng, self.cs, self.L
        m = 15 * cs
        self.y0, self.y1 = max(0, self.ly - m), min(H, self.ly + self.lh + m)
        self.x0, self.x1 = max(0, self.lx - m), min(W, self.lx + self.lw + m)
        c = (slice(self.y0, self.y1), slice(self.x0, self.x1))
        self.c = c
        self.Lc = L[c]
        self.cy, self.cx = self.yy[c], self.xx[c]
        self.cbay = self.bay[c]
        self.crel = self.rel[c]
        self.ctop = self.top[c]
        self.cout = self.out[c]
        h, w = self.Lc.shape
        # night sky: a scrolling strip (2w wide) of stars and a faint diagonal milky way
        tw = 2 * w
        self.stw = tw
        n = max(40, int(L.sum()) * 2 // 21)
        self.sy = r.integers(0, h, n)
        self.sx = r.integers(0, tw, n)
        self.sph = r.uniform(0, 6.28, n)
        self.sfq = r.uniform(0.4, 1.1, n)
        self.sbig = r.random(n) < 0.12
        self.sval = r.uniform(0.45, 1.0, n)
        yy, xx = np.mgrid[0:h, 0:tw].astype(float)
        band = np.sin((xx / tw) * 2 * np.pi) * h * 0.22 + h * 0.5 - (yy)
        nz = box_blur(np.tile(r.random((h, w)), (1, 2)), 2 * cs)
        nz = (nz - nz.mean()) / (nz.std() + 1e-6)
        mw = np.exp(-(band / (h * 0.16)) ** 2) * (0.75 + 0.35 * nz)
        mw = np.clip(mw, 0, 1)
        self.mway = np.concatenate([mw, mw], 1)
        # extra faint dust stars concentrated in the band
        k = n
        dy, dx = r.integers(0, h, k * 3), r.integers(0, tw, k * 3)
        keep = r.random(k * 3) < mw[dy, dx] * 0.8
        self.dy, self.dx = dy[keep], dx[keep]
        # clouds: two tileable layers of hand-built cumulus (overlapping puffs on a flat base)
        self.layers = [self.make_clouds(w, h, 0.55, 3, 0.0), self.make_clouds(w, h, 0.78, 3, 1.0)]

    def make_clouds(self, w, h, size, n, yb):
        """A tileable strip (2w wide, doubled for scrolling): 0 sky, 1 body, 2 lit rim, 3 shade, 4 lobe line."""
        r, cs, lh = self.rng, self.cs, self.lh
        tw = 2 * w
        lab = np.zeros((h, tw), np.int8)
        yy, xx = np.mgrid[0:h, 0:tw].astype(float)
        bay = tile(bayer8(), h, tw)
        for i in range(n):
            cx = (i + r.uniform(0.2, 0.8)) * tw / n
            base = self.ly - self.y0 + lh * (r.uniform(0.45, 0.8) * yb + 0.32 * (1 - yb))
            k = int(r.integers(3, 6))
            width = lh * size * (0.45 + 0.32 * k) * r.uniform(0.9, 1.1)
            puffs = []
            for j in range(k):
                f = (j + 0.5) / k
                pr = lh * size * (0.15 + 0.22 * math.sin(math.pi * f) ** 0.7) * r.uniform(0.85, 1.15)
                puffs.append((pr, cx + (f - 0.5) * width, base - pr * r.uniform(0.6, 0.9)))
            m = np.zeros((h, tw), bool)
            lobe = np.zeros((h, tw), bool)
            for pr, px, py in sorted(puffs):            # paint small puffs first, the big ones in front
                for o in (-tw, 0, tw):
                    d = np.hypot(xx - px - o, (yy - py) * 1.1)
                    inside = (d < pr) & (yy <= base)
                    rim = inside & (d > pr - 0.8 * cs) & (yy < py - pr * 0.2)
                    lobe = (lobe & ~inside) | (rim & m)
                    m |= inside
            top = min(py - pr for pr, px, py in puffs)
            hgt = np.clip((yy - top) / max(base - top, 1), 0, 1)
            edge = lobe
            lab[m] = 1
            lab[m & edge] = 4
            lab[m & (hgt + (bay - 0.5) * 0.16 > 0.72)] = 3
            lab[m & ~shift(m, cs, 0)] = 2
            lab[m & ~shift(m, 0, cs) & (hgt < 0.6)] = 2
        return np.concatenate([lab, lab], 1)

    def draw(self, dt):
        t, cs, L = self.t, self.cs, self.L
        Lc, X, Y, bay = self.Lc, self.cx, self.cy, self.cbay
        p = (t / self.CYCLE + self.P0) % 1.0
        stops, clit, cshade, cover, stars, warm = day_interp(p)

        # --- sky: a banded, ordered-dithered ramp through the five stops
        bands = np.array([stops[min(int(f), 3)] * (1 - (f - min(int(f), 3))) + stops[min(int(f) + 1, 4)] * (f - min(int(f), 3))
                          for f in np.linspace(0, 4, DAY_NB)])
        sky = bands[dith(self.crel, DAY_NB, bay, 0.5)]

        # --- sun: rises bottom-left, arcs over the word, sets bottom-right
        u = (p - 0.135) / 0.545
        sun_alt, sunlum, moonlum = 0.0, None, None
        if -0.06 < u < 1.06:
            uc = min(max(u, 0.0), 1.0)
            alt = math.sin(math.pi * uc)
            sun_alt = alt
            sx = self.lx + self.lw * (0.04 + 0.92 * u)
            sy = self.ly + self.lh * (1.12 - 1.0 * alt)
            low = (1 - alt) ** 1.5
            d = np.hypot(X - sx, (Y - sy) * 1.1)
            R = max(2.6, self.lh * (0.17 + 0.05 * low))
            glowc = np.array([255, 214, 140]) * (1 - low) + np.array([255, 120, 50]) * low
            # wide horizon glow under a low sun
            hz = np.exp(-((X - sx) / (self.lw * 0.35)) ** 2) * self.crel ** 1.6 * warm
            g = np.exp(-np.maximum(d - R, 0) / (self.lh * 0.5)) * (0.55 + 0.25 * low) + hz * 0.55
            sky = sky + qd(np.clip(g, 0, 1), 6, bay, 0.6)[..., None] * (glowc - sky * 0.55)
            core = np.array([255, 252, 236]) * (1 - low) + np.array([255, 190, 96]) * low
            rimc = np.array([255, 232, 150]) * (1 - low) + np.array([255, 128, 56]) * low
            ring = d < R + 1.2 * cs
            sky[ring] = rimc
            sky[d < R] = core
            sunlum = np.exp(-d / (self.lh * 0.7))
            suncol = glowc * (0.3 + 0.35 * low) * min(1.0, (0.56 - abs(u - 0.5)) * 6)
        # --- stars and the milky way
        if stars > 0.02:
            h, w = Lc.shape
            off = int(t * 0.5 * cs) % self.stw            # the sky turns, very slowly
            mw = self.mway[:, self.stw - off:self.stw - off + w]
            sky = sky + (qd(mw * stars, 4, bay, 0.7) * 0.85)[..., None] * np.array([46, 40, 84])
            dx = (self.dx + off) % self.stw
            ok = dx < w
            sky[self.dy[ok], dx[ok]] += np.array([60, 56, 90]) * stars
            sx = (self.sx + off) % self.stw
            ok = sx < w
            sy, sx = self.sy[ok], sx[ok]
            tw = 0.55 + 0.45 * np.sin(t * self.sfq[ok] + self.sph[ok])
            a = (stars * self.sval[ok] * tw)[:, None]
            sc = np.array([240, 240, 255])
            sky[sy, sx] = sky[sy, sx] * (1 - a) + sc * a
            if cs > 1:
                b = self.sbig[ok]
                for oy, ox in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    yy, xx = sy[b] + oy, sx[b] + ox
                    k = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
                    ab = a[b][k] * 0.5
                    sky[yy[k], xx[k]] = sky[yy[k], xx[k]] * (1 - ab) + sc * ab

        # --- the moon, in front of the stars
        v = ((p - 0.70) % 1.0) / 0.42
        if 0 < v < 1:
            alt = math.sin(math.pi * v)
            mx = self.lx + self.lw * (0.06 + 0.88 * v)
            my = self.ly + self.lh * (1.0 - 0.8 * alt)
            vis = smoothstep(v, 0.0, 0.1) * (1 - smoothstep(v, 0.9, 1.0))
            d = np.hypot(X - mx, Y - my)
            R = max(2.4, self.lh * 0.17)
            moonlum = np.exp(-d / (self.lh * 0.6)) * vis
            mg = np.exp(-np.maximum(d - R, 0) / (self.lh * 0.35)) * 0.35 * vis
            sky = sky + qd(mg, 5, bay, 0.6)[..., None] * np.array([140, 150, 200])
            disk = d < R
            dark = np.hypot(X - mx - R * 0.55, Y - my + R * 0.3) < R * 0.9
            sky[disk & ~dark] = sky[disk & ~dark] * (1 - vis) + np.array([240, 238, 214]) * vis
            sky[disk & dark] = sky[disk & dark] * (1 - 0.5 * vis) + np.array([40, 44, 78]) * 0.5 * vis
        # --- clouds drifting slowly left to right, near layer a little faster
        h, w = Lc.shape
        body = clit * 0.6 + cshade * 0.4
        pal = np.array([[0, 0, 0], body, clit, cshade, cshade * 0.6 + body * 0.4])
        for lab, spd in ((self.layers[0], 0.9), (self.layers[1], 1.6)):
            tw = lab.shape[1] // 2
            off = int(t * spd * cs + 37 * spd) % tw
            cl = lab[:, tw - off:tw - off + w]
            m = cl > 0
            sky[m] = pal[cl[m]]
            if sunlum is not None:          # clouds near the sun catch its light
                lit = m & (cl != 3)
                sky[lit] += qd(sunlum[lit], 4, bay[lit], 0.5)[:, None] * (glowc - sky[lit]) * 0.7

        # --- compose
        img = self.bgimg.copy()
        ci = img[self.c]
        glow = stops[1] * 0.4 + stops[2] * 0.6
        glow = np.clip(glow.mean() + (glow - glow.mean()) * 1.7, 0, 255)
        ci += self.halo_c(Lc * 1.0, glow * 0.36, 3, 7, 0.75)
        if sunlum is not None:
            ci += self.halo_c(Lc * sunlum, suncol, 3, 9, 1.2)
        if moonlum is not None:
            ci += self.halo_c(Lc * moonlum, (70, 80, 120), 3, 9, 1.2)
        ci[self.cout] *= 0.25
        fill = sky
        fill[self.ctop] = fill[self.ctop] * 0.8 + 255 * 0.2
        ci[Lc] = fill[Lc]
        return img

    def halo_c(self, lum, col, r1=3, r2=8, k=1.0):
        cs = self.cs
        g = box_blur(lum, r1 * cs)
        g = box_blur(g, r1 * cs) * 1.5 + box_blur(g, r2 * cs) * 0.8
        return qd(np.clip(g * k, 0, 1), 6, self.cbay, 0.5)[..., None] * np.array(col)


SCENES = [Embers, Daylight, Cosmos, Ink, Tide, Lava, Warp, Glitch]
BY_NAME = {c.name: c for c in SCENES}


# --------------------------------------------------------------------------- director / transitions


class Director:
    def __init__(self, W, H, logo, names, duration=24.0, trans=1.1, seed=None):
        self.W, self.H, self.logo, self.names = W, H, logo, names
        self.duration, self.trans = duration, trans
        self.seed = seed if seed is not None else int(time.time())
        self.i, self.k = 0, 0
        self.cur = self.make(0)
        self.next = None
        self.tt = 0.0
        self.bay = tile(bayer8(), H, W)
        yy, xx = np.mgrid[0:H, 0:W]
        self.xx, self.yy = xx, yy

    def make(self, i):
        self.k += 1
        return BY_NAME[self.names[i % len(self.names)]](self.W, self.H, self.logo, self.seed + self.k)

    def skip(self, d):
        self.i = (self.i + d) % len(self.names)
        self.cur, self.next, self.tt = self.make(self.i), None, 0.0

    def frame(self, dt):
        self.tt += dt
        a = self.cur.frame(dt)
        if len(self.names) > 1 and self.next is None and self.tt > self.duration:
            self.next = self.make(self.i + 1)
        if self.next is None:
            return a
        b = self.next.frame(dt)
        p = (self.tt - self.duration) / self.trans
        if p >= 1:
            self.i = (self.i + 1) % len(self.names)
            self.cur, self.next, self.tt = self.next, None, 0.0
            return b
        style = self.i % 3
        if style == 0:  # ordered-dither dissolve
            m = self.bay < p
            return np.where(m[..., None], b, a)
        if style == 1:  # scanline wipe with a hot beam
            edge = p * (self.H + 6)
            m = self.yy < edge
            out = np.where(m[..., None], b, a)
            beam = np.abs(self.yy - edge) < 1.5
            out[beam] = (255, 255, 255)
            return out
        # pixelate out / in
        q = 1 - abs(2 * p - 1)
        bs = 1 + int(q * 12)
        src = a if p < 0.5 else b
        ys = (self.yy // bs) * bs
        xs = (self.xx // bs) * bs
        return src[ys, xs]


# --------------------------------------------------------------------------- terminal output

_FG, _BG = {}, {}


def _fg(c):
    s = _FG.get(c)
    if s is None:
        s = _FG[c] = f"\x1b[38;2;{c >> 16};{(c >> 8) & 255};{c & 255}m"
    return s


def _bg(c):
    s = _BG.get(c)
    if s is None:
        s = _BG[c] = f"\x1b[48;2;{c >> 16};{(c >> 8) & 255};{c & 255}m"
    return s


def encode(frame, prev):
    if len(_FG) > 60000:
        _FG.clear()
        _BG.clear()
    f = frame.astype(np.uint32)
    packed = (f[..., 0] << 16) | (f[..., 1] << 8) | f[..., 2]
    top, bot = packed[0::2], packed[1::2]
    if prev is None or prev[0].shape != top.shape:
        ch = np.ones(top.shape, bool)
    else:
        ch = (top != prev[0]) | (bot != prev[1])
    rows, cols = np.nonzero(ch)
    out = []
    lf = lb = None
    er = ec = -1
    for r, c, t, b in zip(rows.tolist(), cols.tolist(), top[rows, cols].tolist(), bot[rows, cols].tolist()):
        if r != er or c != ec:
            out.append(f"\x1b[{r + 1};{c + 1}H")
        if t == b:
            if b != lb:
                out.append(_bg(b))
                lb = b
            out.append(" ")
        else:
            if t != lf:
                out.append(_fg(t))
                lf = t
            if b != lb:
                out.append(_bg(b))
                lb = b
            out.append("▀")
        er, ec = r, c + 1
    return "".join(out), (top, bot)


def save_png(frame, path, px=6):
    from PIL import Image
    big = np.repeat(np.repeat(frame, px, 0), px, 1)
    Image.fromarray(big).save(path)
    return path


def hypr_focus_lost():
    try:
        out = subprocess.run(["hyprctl", "activewindow", "-j"], capture_output=True, text=True, timeout=0.5).stdout
        return '"org.omarchy.screensaver"' not in out
    except Exception:
        return False


def run_live(args, logo, names):
    import termios
    import tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    out = sys.stdout.buffer
    resized = [True]
    signal.signal(signal.SIGWINCH, lambda *_: resized.__setitem__(0, True))
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    tty.setcbreak(fd)
    out.write(b"\x1b[?1049h\x1b[?25l\x1b[?7l\x1b[0m\x1b[2J")
    out.flush()
    director = None
    prev = None
    frame = None
    last = time.monotonic()
    focus_check = last + 1.5
    try:
        while True:
            if resized[0]:
                resized[0] = False
                cols, rows = os.get_terminal_size()
                i = director.i if director else 0
                director = Director(cols, rows * 2, logo, names, args.duration)
                director.i = i
                director.cur = director.make(i)
                prev = None
                out.write(b"\x1b[0m\x1b[2J")
            now = time.monotonic()
            dt = min(now - last, 0.1)
            last = now
            frame = director.frame(dt)
            s, prev = encode(frame, prev)
            out.write(b"\x1b[?2026h" + s.encode() + b"\x1b[0m\x1b[?2026l")
            out.flush()
            while select.select([sys.stdin], [], [], 0)[0]:
                ch = os.read(fd, 32)
                if ch in (b"n", b"\x1b[C", b" "):
                    director.skip(1)
                    prev = None
                elif ch in (b"p", b"\x1b[D"):
                    director.skip(-1)
                    prev = None
                elif ch == b"s":
                    d = os.path.expanduser("~/Pictures/pixelsaver")
                    os.makedirs(d, exist_ok=True)
                    save_png(frame, os.path.join(d, f"{director.cur.name}-{time.strftime('%Y%m%d-%H%M%S')}.png"))
                else:
                    return
            if args.omarchy and now > focus_check:
                focus_check = now + 1.0
                if hypr_focus_lost():
                    return
            spare = 1 / FPS - (time.monotonic() - now)
            if spare > 0:
                time.sleep(spare)
    except KeyboardInterrupt:
        pass
    finally:
        out.write(b"\x1b[0m\x1b[?7h\x1b[?25h\x1b[?1049l")
        out.flush()
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


# --------------------------------------------------------------------------- offline rendering


def simulate(name, times, W, H, logo, seed=1):
    sc = BY_NAME[name](W, H, logo, seed)
    frames, t = [], 0.0
    for target in sorted(times):
        while t + 1e-9 < target:
            f = sc.frame(DT)
            t += DT
        frames.append(f if target > 0 else sc.frame(DT))
    return frames


def main():
    ap = argparse.ArgumentParser(description="Pixel-art terminal screensaver")
    ap.add_argument("--file", default=DEFAULT_LOGO, help="ASCII block-art file to draw")
    ap.add_argument("--scene", help="comma-separated scenes to cycle (default: all)")
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--duration", type=float, default=24.0, help="seconds per scene")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--omarchy", action="store_true", help="exit when the screensaver window loses focus")
    ap.add_argument("--shot", metavar="SCENE", help="render one PNG headlessly")
    ap.add_argument("--at", type=float, default=6.0)
    ap.add_argument("--sheet", metavar="PNG", help="render a contact sheet of all scenes")
    ap.add_argument("--times", default="2,6,12")
    ap.add_argument("--gif", metavar="SCENE")
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--size", default="160x45", help="COLSxROWS for offline rendering")
    ap.add_argument("--px", type=int, default=6, help="screen pixels per art pixel in PNGs")
    ap.add_argument("--out")
    args = ap.parse_args()

    logo = load_logo(args.file)
    names = args.scene.split(",") if args.scene else [c.name for c in SCENES]
    if args.list:
        for c in SCENES:
            print(f"{c.name:10s} {c.title}")
        return
    cols, rows = (int(v) for v in args.size.lower().split("x"))
    W, H = cols, rows * 2

    if args.shot:
        out = args.out or f"{args.shot}-{args.at:g}s.png"
        save_png(simulate(args.shot, [args.at], W, H, logo)[0], out, args.px)
        print(out)
        return
    if args.gif:
        from PIL import Image
        step = 2
        n = int(args.seconds * FPS)
        times = [i * DT for i in range(0, n, step)]
        frames = simulate(args.gif, times, W, H, logo)
        imgs = [Image.fromarray(np.repeat(np.repeat(f, args.px, 0), args.px, 1)) for f in frames]
        out = args.out or f"{args.gif}.gif"
        imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=int(1000 * DT * step), loop=0)
        print(out)
        return
    if args.sheet:
        from PIL import Image
        times = [float(x) for x in args.times.split(",")]
        gap, lab = 6, 10
        tiles = []
        for nm in names:
            fr = simulate(nm, times, W, H, logo)
            row = []
            for f in fr:
                row.append(np.repeat(np.repeat(f, args.px, 0), args.px, 1))
                row.append(np.full((H * args.px, gap * args.px, 3), 30, np.uint8))
            label = np.full((lab * args.px, (W + gap) * len(fr) * args.px, 3), 30, np.uint8)
            lm = text_mask(f"{nm.upper()} - {BY_NAME[nm].title}   T={','.join(f'{x:g}' for x in times)}S", 1)
            lm = np.kron(lm, np.ones((args.px, args.px), bool))
            blit(label, lm, 2 * args.px, 2 * args.px, (230, 230, 230))
            tiles.append(np.vstack([label, np.hstack(row)]))
        Image.fromarray(np.vstack(tiles)).save(args.sheet)
        print(args.sheet)
        return
    if args.shuffle:
        np.random.default_rng().shuffle(names)
    if not sys.stdin.isatty():
        sys.exit("pixelsaver: live mode needs a terminal (use --shot/--sheet/--gif offline)")
    run_live(args, logo, names)


if __name__ == "__main__":
    main()
