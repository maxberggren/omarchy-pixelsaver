//! Terminal plumbing with no dependencies: termios via libc symbols, window size, signals,
//! a ppoll-based frame wait, and a diffing half-block encoder.
#![allow(non_camel_case_types)]

use std::sync::atomic::{AtomicBool, Ordering};

#[repr(C)]
#[derive(Clone, Copy)]
pub struct termios {
    c_iflag: u32,
    c_oflag: u32,
    c_cflag: u32,
    c_lflag: u32,
    c_line: u8,
    c_cc: [u8; 32],
    c_ispeed: u32,
    c_ospeed: u32,
}

#[repr(C)]
struct winsize {
    ws_row: u16,
    ws_col: u16,
    ws_xpixel: u16,
    ws_ypixel: u16,
}

#[repr(C)]
struct pollfd {
    fd: i32,
    events: i16,
    revents: i16,
}

#[repr(C)]
struct timespec {
    tv_sec: i64,
    tv_nsec: i64,
}

extern "C" {
    fn tcgetattr(fd: i32, t: *mut termios) -> i32;
    fn tcsetattr(fd: i32, act: i32, t: *const termios) -> i32;
    fn ioctl(fd: i32, req: u64, ...) -> i32;
    fn ppoll(fds: *mut pollfd, n: u64, ts: *const timespec, mask: *const u8) -> i32;
    fn read(fd: i32, buf: *mut u8, n: usize) -> isize;
    fn write(fd: i32, buf: *const u8, n: usize) -> isize;
    fn signal(sig: i32, handler: usize) -> usize;
    fn isatty(fd: i32) -> i32;
}

const ICANON: u32 = 0o2;
const ECHO: u32 = 0o10;
const VMIN: usize = 6;
const VTIME: usize = 5;
const TCSADRAIN: i32 = 1;
const TIOCGWINSZ: u64 = 0x5413;
const POLLIN: i16 = 1;
const SIGHUP: i32 = 1;
const SIGINT: i32 = 2;
const SIGTERM: i32 = 15;
const SIGWINCH: i32 = 28;

pub static RESIZED: AtomicBool = AtomicBool::new(true);
pub static QUIT: AtomicBool = AtomicBool::new(false);

extern "C" fn on_winch(_: i32) {
    RESIZED.store(true, Ordering::Relaxed);
}
extern "C" fn on_quit(_: i32) {
    QUIT.store(true, Ordering::Relaxed);
}

pub fn is_tty(fd: i32) -> bool {
    unsafe { isatty(fd) == 1 }
}

/// Puts the terminal in cbreak mode on construction and restores it (and the screen) on drop.
pub struct Term {
    old: termios,
}

impl Term {
    pub fn enter() -> Option<Term> {
        unsafe {
            let mut t: termios = std::mem::zeroed();
            if tcgetattr(0, &mut t) != 0 {
                return None;
            }
            let old = t;
            t.c_lflag &= !(ICANON | ECHO);
            t.c_cc[VMIN] = 0;
            t.c_cc[VTIME] = 0;
            tcsetattr(0, TCSADRAIN, &t);
            signal(SIGWINCH, on_winch as extern "C" fn(i32) as usize);
            signal(SIGINT, on_quit as extern "C" fn(i32) as usize);
            signal(SIGTERM, on_quit as extern "C" fn(i32) as usize);
            signal(SIGHUP, on_quit as extern "C" fn(i32) as usize);
            out(b"\x1b[?1049h\x1b[?25l\x1b[?7l\x1b[0m\x1b[2J");
            Some(Term { old })
        }
    }
}

impl Drop for Term {
    fn drop(&mut self) {
        out(b"\x1b[0m\x1b[?7h\x1b[?25h\x1b[?1049l");
        unsafe {
            tcsetattr(0, TCSADRAIN, &self.old);
        }
    }
}

/// Terminal size as (cols, rows).
pub fn size() -> (usize, usize) {
    unsafe {
        let mut ws: winsize = std::mem::zeroed();
        if ioctl(1, TIOCGWINSZ, &mut ws as *mut winsize) == 0 && ws.ws_col > 0 && ws.ws_row > 0 {
            (ws.ws_col as usize, ws.ws_row as usize)
        } else {
            (80, 24)
        }
    }
}

/// Write everything to stdout (fd 1), retrying on partial writes.
pub fn out(mut b: &[u8]) {
    while !b.is_empty() {
        let n = unsafe { write(1, b.as_ptr(), b.len()) };
        if n <= 0 {
            if std::io::Error::last_os_error().kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            return;
        }
        b = &b[n as usize..];
    }
}

/// Sleep up to `ns` nanoseconds, waking early if stdin becomes readable or a signal arrives.
/// Returns any bytes read from stdin.
pub fn wait_input(ns: i64, buf: &mut [u8]) -> usize {
    unsafe {
        let mut p = pollfd { fd: 0, events: POLLIN, revents: 0 };
        let ts = timespec { tv_sec: ns.max(0) / 1_000_000_000, tv_nsec: ns.max(0) % 1_000_000_000 };
        let r = ppoll(&mut p, 1, &ts, std::ptr::null());
        if r > 0 && (p.revents & POLLIN) != 0 {
            let n = read(0, buf.as_mut_ptr(), buf.len());
            if n > 0 {
                return n as usize;
            }
            if n == 0 {
                QUIT.store(true, Ordering::Relaxed); // stdin closed
            }
        }
        0
    }
}

// --------------------------------------------------------------------------- encoder

/// Turns RGB frames into minimal escape-code updates: one cell = two vertical pixels ("▀" with fg = top,
/// bg = bottom), only changed cells are sent, cursor moves and colour changes are elided when possible.
pub struct Encoder {
    top: Vec<u32>,
    bot: Vec<u32>,
    cols: usize,
    valid: bool,
    pub buf: Vec<u8>,
}

#[inline]
fn push_num(b: &mut Vec<u8>, mut n: u32) {
    let mut tmp = [0u8; 10];
    let mut i = tmp.len();
    loop {
        i -= 1;
        tmp[i] = b'0' + (n % 10) as u8;
        n /= 10;
        if n == 0 {
            break;
        }
    }
    b.extend_from_slice(&tmp[i..]);
}

#[inline]
fn push_rgb(b: &mut Vec<u8>, c: u32) {
    push_num(b, c >> 16);
    b.push(b';');
    push_num(b, (c >> 8) & 255);
    b.push(b';');
    push_num(b, c & 255);
    b.push(b'm');
}

impl Encoder {
    pub fn new() -> Self {
        Encoder { top: vec![], bot: vec![], cols: 0, valid: false, buf: Vec::with_capacity(1 << 16) }
    }
    pub fn invalidate(&mut self) {
        self.valid = false;
    }
    /// Encode a w x h frame (h even). Returns the bytes to write (empty if nothing changed).
    pub fn encode(&mut self, f: &[[u8; 3]], w: usize, h: usize) -> &[u8] {
        let rows = h / 2;
        if !self.valid || self.cols != w || self.top.len() != w * rows {
            self.top = vec![u32::MAX; w * rows];
            self.bot = vec![u32::MAX; w * rows];
            self.cols = w;
            self.valid = true;
        }
        let b = &mut self.buf;
        b.clear();
        let (mut lf, mut lb) = (u32::MAX, u32::MAX);
        let (mut er, mut ec) = (usize::MAX, usize::MAX);
        let pack = |p: [u8; 3]| ((p[0] as u32) << 16) | ((p[1] as u32) << 8) | p[2] as u32;
        for r in 0..rows {
            let rt = &f[2 * r * w..2 * r * w + w];
            let rb = &f[(2 * r + 1) * w..(2 * r + 1) * w + w];
            for c in 0..w {
                let (t, bt) = (pack(rt[c]), pack(rb[c]));
                let k = r * w + c;
                if self.top[k] == t && self.bot[k] == bt {
                    continue;
                }
                self.top[k] = t;
                self.bot[k] = bt;
                if r != er || c != ec {
                    b.extend_from_slice(b"\x1b[");
                    push_num(b, r as u32 + 1);
                    b.push(b';');
                    push_num(b, c as u32 + 1);
                    b.push(b'H');
                }
                if t == bt {
                    if bt != lb {
                        b.extend_from_slice(b"\x1b[48;2;");
                        push_rgb(b, bt);
                        lb = bt;
                    }
                    b.push(b' ');
                } else {
                    if t != lf {
                        b.extend_from_slice(b"\x1b[38;2;");
                        push_rgb(b, t);
                        lf = t;
                    }
                    if bt != lb {
                        b.extend_from_slice(b"\x1b[48;2;");
                        push_rgb(b, bt);
                        lb = bt;
                    }
                    b.extend_from_slice("▀".as_bytes());
                }
                er = r;
                ec = c + 1;
            }
        }
        &self.buf
    }
}
