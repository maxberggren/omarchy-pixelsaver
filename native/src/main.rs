//! pixelsaver (native): the pixel-art terminal screensaver, compiled.
//!
//!   pixelsaver                         cycle all scenes (n/p or arrows = next/prev, s = screenshot, other keys quit)
//!   pixelsaver --scene embers,ink      cycle some scenes        --duration 24   --fps 30   --shuffle
//!   pixelsaver --omarchy               also exit when the screensaver window loses focus (Hyprland)
//!   pixelsaver --file art.txt          draw any block-art file instead of the Omarchy logo
//!   pixelsaver --shot embers --at 2,6,12 --size 200x56 --px 2 --out strip.png
//!   pixelsaver --bench all --seconds 40 --size 200x56
//!   pixelsaver --list
mod core;
mod director;
mod font;
mod png;
mod scenes;
mod term;

use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use crate::core::*;
use director::Director;

struct Args {
    file: String,
    scenes: Vec<String>,
    shuffle: bool,
    duration: f32,
    fps: f32,
    omarchy: bool,
    list: bool,
    shot: Option<String>,
    at: Vec<f32>,
    bench: Option<String>,
    seconds: f32,
    size: (usize, usize),
    px: usize,
    out: Option<String>,
    seed: Option<u64>,
}

fn usage() -> ! {
    eprintln!(
        "usage: pixelsaver [--scene a,b] [--shuffle] [--duration S] [--fps N] [--omarchy] [--file ART]\n\
         \x20      pixelsaver --shot NAME [--at 2,6] [--size COLSxROWS] [--px N] [--out F.png]\n\
         \x20      pixelsaver --bench NAME|all [--seconds S] [--size COLSxROWS]\n\
         \x20      pixelsaver --list"
    );
    std::process::exit(2)
}

fn parse() -> Args {
    let home = std::env::var("HOME").unwrap_or_default();
    let mut a = Args {
        file: format!("{home}/.config/omarchy/branding/screensaver.txt"),
        scenes: scenes::NAMES.iter().map(|n| n.0.to_string()).collect(),
        shuffle: false,
        duration: 24.0,
        fps: 30.0,
        omarchy: false,
        list: false,
        shot: None,
        at: vec![6.0],
        bench: None,
        seconds: 30.0,
        size: (160, 45),
        px: 6,
        out: None,
        seed: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        let mut v = || it.next().unwrap_or_else(|| usage());
        match k.as_str() {
            "--file" => a.file = v(),
            "--scene" => a.scenes = v().split(',').map(String::from).collect(),
            "--shuffle" => a.shuffle = true,
            "--duration" => a.duration = v().parse().unwrap_or_else(|_| usage()),
            "--fps" => a.fps = v().parse::<f32>().unwrap_or_else(|_| usage()).clamp(1.0, 120.0),
            "--omarchy" => a.omarchy = true,
            "--list" => a.list = true,
            "--shot" => a.shot = Some(v()),
            "--at" => a.at = v().split(',').map(|x| x.parse().unwrap_or_else(|_| usage())).collect(),
            "--bench" => a.bench = Some(v()),
            "--seconds" => a.seconds = v().parse().unwrap_or_else(|_| usage()),
            "--size" => {
                let s = v();
                let (c, r) = s.split_once('x').unwrap_or_else(|| usage());
                a.size = (c.parse().unwrap_or_else(|_| usage()), r.parse().unwrap_or_else(|_| usage()));
            }
            "--px" => a.px = v().parse().unwrap_or_else(|_| usage()),
            "--out" => a.out = Some(v()),
            "--seed" => a.seed = Some(v().parse().unwrap_or_else(|_| usage())),
            "-h" | "--help" => usage(),
            _ => usage(),
        }
    }
    for s in &a.scenes {
        if !scenes::NAMES.iter().any(|n| n.0 == s) {
            eprintln!("pixelsaver: unknown scene '{s}' (see --list)");
            std::process::exit(2);
        }
    }
    a
}

fn now_seed() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(1)
}

/// Run a scene headlessly at the fixed 30 fps step and capture frames at the given times.
fn simulate(name: &str, times: &[f32], w: usize, h: usize, logo: &Logo, seed: u64) -> Vec<Vec<[u8; 3]>> {
    let dt = 1.0 / 30.0;
    let mut sc = scenes::make(name, w, h, logo, seed).unwrap();
    let mut c = Canvas::new(w, h, [0.0; 3]);
    let mut ts = times.to_vec();
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut out = vec![];
    let mut t = 0.0f32;
    let mut f = vec![[0u8; 3]; w * h];
    let mut drawn = false;
    for target in ts {
        while t + 1e-6 < target || !drawn {
            sc.draw(dt, &mut c);
            t += dt;
            drawn = true;
        }
        c.to_u8(&mut f);
        out.push(f.clone());
    }
    out
}

fn shot(a: &Args, logo: &Logo) {
    let name = a.shot.as_deref().unwrap();
    let (w, h) = (a.size.0, a.size.1 * 2);
    let frames = simulate(name, &a.at, w, h, logo, a.seed.unwrap_or(1));
    let gap = 4;
    let sw = frames.len() * (w + gap) - gap;
    let mut strip = vec![[40u8; 3]; sw * h];
    for (i, f) in frames.iter().enumerate() {
        for y in 0..h {
            let o = y * sw + i * (w + gap);
            strip[o..o + w].copy_from_slice(&f[y * w..(y + 1) * w]);
        }
    }
    let path = a.out.clone().unwrap_or_else(|| format!("{name}.png"));
    std::fs::write(&path, png::encode(&strip, sw, h, a.px.max(1))).expect("write png");
    println!("{path}");
}

fn bench(a: &Args, logo: &Logo) {
    let which = a.bench.as_deref().unwrap();
    let names: Vec<&str> =
        if which == "all" { scenes::NAMES.iter().map(|n| n.0).collect() } else { which.split(',').collect() };
    let (w, h) = (a.size.0, a.size.1 * 2);
    println!("{w}x{h} px ({}x{} cells), {} s per scene, draw+encode per frame:", a.size.0, a.size.1, a.seconds);
    for name in names {
        let t0 = Instant::now();
        let Some(mut sc) = scenes::make(name, w, h, logo, a.seed.unwrap_or(1)) else {
            eprintln!("unknown scene {name}");
            continue;
        };
        let init = t0.elapsed();
        let mut c = Canvas::new(w, h, [0.0; 3]);
        let mut f = vec![[0u8; 3]; w * h];
        let mut enc = term::Encoder::new();
        let n = (a.seconds * 30.0) as usize;
        let mut ts = Vec::with_capacity(n);
        let mut bytes = 0usize;
        for _ in 0..n {
            let s = Instant::now();
            sc.draw(1.0 / 30.0, &mut c);
            c.to_u8(&mut f);
            bytes += enc.encode(&f, w, h).len();
            ts.push(s.elapsed().as_secs_f64() * 1000.0);
        }
        let avg = ts.iter().sum::<f64>() / n as f64;
        ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "  {name:9} init {:6.2} ms  avg {:6.3} ms  p95 {:6.3} ms  max {:6.3} ms  ~{:.0} KB/s output  ({:.1}% of one core at 30 fps)",
            init.as_secs_f64() * 1000.0,
            avg,
            ts[n * 95 / 100],
            ts[n - 1],
            bytes as f64 / a.seconds as f64 / 1024.0,
            avg * 30.0 / 10.0
        );
    }
}

/// True if Hyprland reports a focused window that is not the screensaver. Talks to the IPC socket directly.
fn hypr_focus_lost() -> bool {
    use std::io::{Read, Write};
    let (Ok(rt), Ok(sig)) = (std::env::var("XDG_RUNTIME_DIR"), std::env::var("HYPRLAND_INSTANCE_SIGNATURE")) else {
        return false;
    };
    let Ok(mut s) = std::os::unix::net::UnixStream::connect(format!("{rt}/hypr/{sig}/.socket.sock")) else {
        return false;
    };
    let _ = s.set_read_timeout(Some(Duration::from_millis(300)));
    if s.write_all(b"j/activewindow").is_err() {
        return false;
    }
    let mut buf = String::new();
    if s.read_to_string(&mut buf).is_err() || buf.trim().is_empty() {
        return false;
    }
    !buf.contains("\"org.omarchy.screensaver\"")
}

fn screenshot(d: &Director) {
    let dir = format!("{}/Pictures/pixelsaver", std::env::var("HOME").unwrap_or_default());
    let _ = std::fs::create_dir_all(&dir);
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|x| x.as_secs()).unwrap_or(0);
    let path = format!("{dir}/{}-{secs}.png", d.names[d.i]);
    let _ = std::fs::write(path, png::encode(&d.frame, d.w, d.h, 6));
}

fn live(a: &Args, logo: &Logo) {
    if !term::is_tty(0) {
        eprintln!("pixelsaver: live mode needs a terminal (use --shot / --bench offline)");
        std::process::exit(1);
    }
    let mut names = a.scenes.clone();
    let seed = a.seed.unwrap_or_else(now_seed);
    if a.shuffle {
        Rng::new(seed ^ 0xA5A5).shuffle(&mut names);
    }
    let Some(_term) = term::Term::enter() else {
        eprintln!("pixelsaver: cannot set terminal mode");
        std::process::exit(1);
    };
    let frame_ns = (1e9 / a.fps as f64) as i64;
    let mut enc = term::Encoder::new();
    let mut dir: Option<Director> = None;
    let mut last = Instant::now();
    let mut next = last;
    let mut focus_check = last + Duration::from_millis(1500);
    let mut inbuf = [0u8; 64];
    let mut outbuf: Vec<u8> = Vec::with_capacity(1 << 16);
    loop {
        if term::QUIT.load(Ordering::Relaxed) {
            break;
        }
        if term::RESIZED.swap(false, Ordering::Relaxed) {
            let (cols, rows) = term::size();
            let i = dir.as_ref().map(|d| d.i).unwrap_or(0);
            dir = Some(Director::new(cols, rows * 2, logo, names.clone(), a.duration, seed, i));
            enc.invalidate();
            term::out(b"\x1b[0m\x1b[2J");
        }
        let d = dir.as_mut().unwrap();
        let now = Instant::now();
        let dt = (now - last).as_secs_f32().min(0.1);
        last = now;
        d.step(dt);
        let bytes = enc.encode(&d.frame, d.w, d.h);
        if !bytes.is_empty() {
            outbuf.clear();
            outbuf.extend_from_slice(b"\x1b[?2026h");
            outbuf.extend_from_slice(bytes);
            outbuf.extend_from_slice(b"\x1b[0m\x1b[?2026l");
            term::out(&outbuf);
        }
        if a.omarchy && now >= focus_check {
            focus_check = now + Duration::from_secs(1);
            if hypr_focus_lost() {
                break;
            }
        }
        // sleep until the next frame, waking for keys / signals
        next += Duration::from_nanos(frame_ns as u64);
        let now = Instant::now();
        if next < now {
            next = now;
        }
        loop {
            let left = next.saturating_duration_since(Instant::now()).as_nanos() as i64;
            let n = term::wait_input(left, &mut inbuf);
            if n > 0 {
                let k = &inbuf[..n];
                match k {
                    b"n" | b" " | b"\x1b[C" => {
                        d.skip(1);
                        enc.invalidate();
                    }
                    b"p" | b"\x1b[D" => {
                        d.skip(-1);
                        enc.invalidate();
                    }
                    b"s" => screenshot(d),
                    _ => return,
                }
            }
            if term::QUIT.load(Ordering::Relaxed) || term::RESIZED.load(Ordering::Relaxed) || left <= 0 || n > 0 {
                break;
            }
            if Instant::now() >= next {
                break;
            }
        }
    }
}

fn main() {
    let a = parse();
    if a.list {
        for (n, t) in scenes::NAMES {
            println!("{n:10} {t}");
        }
        return;
    }
    let logo = load_logo(&a.file);
    if a.shot.is_some() {
        shot(&a, &logo);
    } else if a.bench.is_some() {
        bench(&a, &logo);
    } else {
        live(&a, &logo);
    }
}
