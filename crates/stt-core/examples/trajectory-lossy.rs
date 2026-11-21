//! Lossy trajectory representations as rate–distortion curves (branch `trajectory`).
//!
//! Question: at an equal, *measured* positional error, is a fitted motion
//! model (cubic Hermite segments, the TRiGS / Animated-SOG idea) smaller than
//! linear keyframes (TD-TR, what `--time-aware-simplify` does), and is either
//! smaller than plain coarser quantization of every vertex?
//!
//! Every method is reconstructed from its *quantized* stored form and its
//! error is measured against every original sample at that sample's own
//! timestamp (synchronized Euclidean distance) — so the reported error is the
//! real error of what would ship, not the fitter's internal tolerance. All
//! methods share one entropy back end: per-trajectory best-of three integer
//! predictors, zigzag varint, zstd-3 over chunks of 500 trajectories.
//!
//! Input: the flat binary written by poopdeck `tools/bench/trajectory-codec/prep.py export`.
//!
//!   cargo run --release -p stt-core --example trajectory-lossy -- <traj.bin>

use rayon::prelude::*;
use std::io::Read;

const M: f64 = 111_320.0;

struct Traj {
    x: Vec<f64>, // local metres
    y: Vec<f64>,
    t: Vec<f64>, // seconds from first fix
    t_ms: Vec<i64>,
}

fn load(path: &str) -> Vec<Traj> {
    let mut buf = Vec::new();
    std::fs::File::open(path)
        .unwrap()
        .read_to_end(&mut buf)
        .unwrap();
    let u = |i: usize| u64::from_le_bytes(buf[i * 8..i * 8 + 8].try_into().unwrap());
    let (nt, nv) = (u(0) as usize, u(1) as usize);
    let offs: Vec<usize> = (0..=nt).map(|i| u(2 + i) as usize).collect();
    let base = 2 + nt + 1;
    let f = |i: usize| f64::from_le_bytes(buf[i * 8..i * 8 + 8].try_into().unwrap());
    let g = |i: usize| i64::from_le_bytes(buf[i * 8..i * 8 + 8].try_into().unwrap());
    (0..nt)
        .map(|k| {
            let (s, e) = (offs[k], offs[k + 1]);
            let lat0 = f(base + nv + s);
            let lon0 = f(base + s);
            let c = lat0.to_radians().cos().max(0.05);
            let t0 = g(base + 2 * nv + s);
            Traj {
                x: (s..e).map(|i| (f(base + i) - lon0) * c * M).collect(),
                y: (s..e).map(|i| (f(base + nv + i) - lat0) * M).collect(),
                t: (s..e)
                    .map(|i| (g(base + 2 * nv + i) - t0) as f64 / 1000.0)
                    .collect(),
                t_ms: (s..e).map(|i| g(base + 2 * nv + i)).collect(),
            }
        })
        .collect()
}

// ---------------------------------------------------------------- entropy back end
fn zigzag(v: i64) -> u64 {
    ((v << 1) ^ (v >> 63)) as u64
}
fn put(buf: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        buf.push((v as u8) | 0x80);
        v >>= 7;
    }
    buf.push(v as u8);
}
fn vlen(v: i64) -> usize {
    (64 - zigzag(v).max(1).leading_zeros()).div_ceil(7) as usize
}

/// Integer series (one per axis, interleaved) + times -> bytes, best of three predictors.
fn code_series(q: &[[i64; 2]], tq: &[i64], out: &mut Vec<u8>) {
    let resid = |p: u8| -> Vec<i64> {
        let mut r = Vec::with_capacity(q.len() * 2);
        for i in 0..q.len() {
            for a in 0..2 {
                let pred = if i == 0 {
                    0
                } else if i == 1 || p == 0 {
                    q[i - 1][a]
                } else {
                    let v = q[i - 1][a] - q[i - 2][a];
                    if p == 1 {
                        q[i - 1][a] + v
                    } else {
                        let (dt0, dt1) = (tq[i - 1] - tq[i - 2], tq[i] - tq[i - 1]);
                        if dt0 > 0 {
                            q[i - 1][a] + ((v as f64) * dt1 as f64 / dt0 as f64).round() as i64
                        } else {
                            q[i - 1][a] + v
                        }
                    }
                };
                r.push(q[i][a] - pred);
            }
        }
        r
    };
    let (p, r) = (0u8..3)
        .map(|p| (p, resid(p)))
        .min_by_key(|(_, r)| r.iter().map(|&v| vlen(v)).sum::<usize>())
        .unwrap();
    out.push(p);
    for v in r {
        put(out, zigzag(v));
    }
    let mut prev = 0;
    for &t in tq {
        put(out, zigzag(t - prev));
        prev = t;
    }
}

/// Time unit: seconds when every fix is on a whole second (AIS, drifters), else ms.
fn time_unit_ms(trajs: &[Traj]) -> i64 {
    if trajs.iter().all(|t| t.t_ms.iter().all(|v| v % 1000 == 0)) {
        1000
    } else {
        1
    }
}

// ---------------------------------------------------------------- error measurement
#[derive(Default, Clone)]
struct Err {
    max: f64,
    sum: f64,
    n: usize,
    hist: Vec<f64>,
}
impl Err {
    fn add(&mut self, e: f64) {
        self.max = self.max.max(e);
        self.sum += e;
        self.n += 1;
        if self.n.is_multiple_of(7) {
            self.hist.push(e);
        }
    }
    fn merge(mut self, o: Err) -> Err {
        self.max = self.max.max(o.max);
        self.sum += o.sum;
        self.n += o.n;
        self.hist.extend(o.hist);
        self
    }
    fn p99(&self) -> f64 {
        let mut h = self.hist.clone();
        h.sort_by(|a, b| a.partial_cmp(b).unwrap());
        h.get(((h.len() as f64) * 0.99) as usize)
            .copied()
            .unwrap_or(0.0)
    }
}

// ---------------------------------------------------------------- representations
/// A shipped trajectory: keyframe indices (into the original samples),
/// quantized positions, optional quantized Hermite tangents (out of a, into b,
/// in grid units, one pair per segment).
struct Shipped {
    keys: Vec<usize>,
    q: Vec<[i64; 2]>,
    tan: Option<Vec<[[i64; 2]; 2]>>,
}

fn qpos(tr: &Traj, i: usize, s: f64) -> [i64; 2] {
    [(tr.x[i] / s).round() as i64, (tr.y[i] / s).round() as i64]
}

fn hermite(p0: [f64; 2], p1: [f64; 2], m0: [f64; 2], m1: [f64; 2], u: f64) -> [f64; 2] {
    let (u2, u3) = (u * u, u * u * u);
    let h00 = 2.0 * u3 - 3.0 * u2 + 1.0;
    let h10 = u3 - 2.0 * u2 + u;
    let h01 = -2.0 * u3 + 3.0 * u2;
    let h11 = u3 - u2;
    [
        h00 * p0[0] + h10 * m0[0] + h01 * p1[0] + h11 * m1[0],
        h00 * p0[1] + h10 * m0[1] + h01 * p1[1] + h11 * m1[1],
    ]
}

/// Reconstruct position at every original sample and accumulate the error.
fn measure(tr: &Traj, sh: &Shipped, s: f64, err: &mut Err) {
    for k in 0..sh.keys.len() {
        let a = sh.keys[k];
        let pa = [sh.q[k][0] as f64 * s, sh.q[k][1] as f64 * s];
        err.add(((pa[0] - tr.x[a]).powi(2) + (pa[1] - tr.y[a]).powi(2)).sqrt());
        if k + 1 == sh.keys.len() {
            break;
        }
        let b = sh.keys[k + 1];
        let pb = [sh.q[k + 1][0] as f64 * s, sh.q[k + 1][1] as f64 * s];
        let dt = tr.t[b] - tr.t[a];
        for i in a + 1..b {
            let u = if dt > 0.0 {
                (tr.t[i] - tr.t[a]) / dt
            } else {
                0.0
            };
            let p = match &sh.tan {
                None => [pa[0] + (pb[0] - pa[0]) * u, pa[1] + (pb[1] - pa[1]) * u],
                Some(tn) => {
                    let m = tn[k];
                    hermite(
                        pa,
                        pb,
                        [m[0][0] as f64 * s, m[0][1] as f64 * s],
                        [m[1][0] as f64 * s, m[1][1] as f64 * s],
                        u,
                    )
                }
            };
            err.add(((p[0] - tr.x[i]).powi(2) + (p[1] - tr.y[i]).powi(2)).sqrt());
        }
    }
}

/// TD-TR: Douglas–Peucker on synchronized Euclidean distance (the
/// `--time-aware-simplify` algorithm), on the unquantized track.
fn tdtr(tr: &Traj, eps: f64) -> Vec<usize> {
    let n = tr.x.len();
    let mut keep = vec![false; n];
    keep[0] = true;
    keep[n - 1] = true;
    let mut stack = vec![(0usize, n - 1)];
    while let Some((a, b)) = stack.pop() {
        if b <= a + 1 {
            continue;
        }
        let dt = tr.t[b] - tr.t[a];
        let (mut worst, mut wi) = (0.0, a);
        for i in a + 1..b {
            let u = if dt > 0.0 {
                (tr.t[i] - tr.t[a]) / dt
            } else {
                0.0
            };
            let px = tr.x[a] + (tr.x[b] - tr.x[a]) * u;
            let py = tr.y[a] + (tr.y[b] - tr.y[a]) * u;
            let d = ((px - tr.x[i]).powi(2) + (py - tr.y[i]).powi(2)).sqrt();
            if d > worst {
                worst = d;
                wi = i;
            }
        }
        if worst > eps {
            keep[wi] = true;
            stack.push((a, wi));
            stack.push((wi, b));
        }
    }
    (0..n).filter(|&i| keep[i]).collect()
}

/// Least-squares Hermite tangents for fixed endpoints over samples a..=b
/// (Schneider-style fit), quantized to the grid; returns the max SED.
fn fit_hermite(tr: &Traj, a: usize, b: usize, s: f64) -> ([[i64; 2]; 2], f64) {
    let pa = qpos(tr, a, s);
    let pb = qpos(tr, b, s);
    let (pa_f, pb_f) = (
        [pa[0] as f64 * s, pa[1] as f64 * s],
        [pb[0] as f64 * s, pb[1] as f64 * s],
    );
    let dt = tr.t[b] - tr.t[a];
    // Normal equations for (m0, m1) per axis: residual r_i = d_i - h10 m0 - h11 m1.
    let (mut a11, mut a12, mut a22) = (0.0, 0.0, 0.0);
    let mut rhs = [[0.0f64; 2]; 2];
    for i in a + 1..b {
        let u = if dt > 0.0 {
            (tr.t[i] - tr.t[a]) / dt
        } else {
            0.0
        };
        let (u2, u3) = (u * u, u * u * u);
        let (h00, h10, h01, h11) = (
            2.0 * u3 - 3.0 * u2 + 1.0,
            u3 - 2.0 * u2 + u,
            -2.0 * u3 + 3.0 * u2,
            u3 - u2,
        );
        a11 += h10 * h10;
        a12 += h10 * h11;
        a22 += h11 * h11;
        for (ax, (p, (q0, q1))) in [tr.x[i], tr.y[i]]
            .iter()
            .zip([(pa_f[0], pb_f[0]), (pa_f[1], pb_f[1])])
            .enumerate()
        {
            let d = p - h00 * q0 - h01 * q1;
            rhs[0][ax] += h10 * d;
            rhs[1][ax] += h11 * d;
        }
    }
    let det = a11 * a22 - a12 * a12;
    let chord = [pb_f[0] - pa_f[0], pb_f[1] - pa_f[1]];
    let (m0, m1) = if det.abs() > 1e-12 {
        let m0 = [
            (a22 * rhs[0][0] - a12 * rhs[1][0]) / det,
            (a22 * rhs[0][1] - a12 * rhs[1][1]) / det,
        ];
        let m1 = [
            (a11 * rhs[1][0] - a12 * rhs[0][0]) / det,
            (a11 * rhs[1][1] - a12 * rhs[0][1]) / det,
        ];
        (m0, m1)
    } else {
        (chord, chord) // too few samples: the linear segment
    };
    let mq = [
        [(m0[0] / s).round() as i64, (m0[1] / s).round() as i64],
        [(m1[0] / s).round() as i64, (m1[1] / s).round() as i64],
    ];
    let (m0q, m1q) = (
        [mq[0][0] as f64 * s, mq[0][1] as f64 * s],
        [mq[1][0] as f64 * s, mq[1][1] as f64 * s],
    );
    let mut worst: f64 = 0.0;
    for i in a + 1..b {
        let u = if dt > 0.0 {
            (tr.t[i] - tr.t[a]) / dt
        } else {
            0.0
        };
        let p = hermite(pa_f, pb_f, m0q, m1q, u);
        worst = worst.max(((p[0] - tr.x[i]).powi(2) + (p[1] - tr.y[i]).powi(2)).sqrt());
    }
    (mq, worst)
}

/// Linear segment error with quantized endpoints.
fn lin_err(tr: &Traj, a: usize, b: usize, s: f64) -> f64 {
    let pa = qpos(tr, a, s);
    let pb = qpos(tr, b, s);
    let dt = tr.t[b] - tr.t[a];
    let mut worst: f64 = 0.0;
    for i in a + 1..b {
        let u = if dt > 0.0 {
            (tr.t[i] - tr.t[a]) / dt
        } else {
            0.0
        };
        let px = (pa[0] as f64 + (pb[0] - pa[0]) as f64 * u) * s;
        let py = (pa[1] as f64 + (pb[1] - pa[1]) as f64 * u) * s;
        worst = worst.max(((px - tr.x[i]).powi(2) + (py - tr.y[i]).powi(2)).sqrt());
    }
    worst
}

const MAX_SPAN: usize = 4096;

/// Greedy segmentation: from each keyframe, exponential then binary search
/// for the farthest end whose segment stays within `eps`.
fn greedy<F: Fn(usize, usize) -> f64>(n: usize, eps: f64, err: F) -> Vec<usize> {
    let mut keys = vec![0usize];
    let mut a = 0usize;
    while a < n - 1 {
        let mut good = a + 1;
        let mut step = 1usize;
        let mut bad = None;
        loop {
            let b = (a + step * 2).min(n - 1).min(a + MAX_SPAN);
            if b == good {
                break;
            }
            if err(a, b) <= eps {
                good = b;
                step *= 2;
            } else {
                bad = Some(b);
                break;
            }
        }
        if let Some(mut hi) = bad {
            while hi > good + 1 {
                let mid = (good + hi) / 2;
                if err(a, mid) <= eps {
                    good = mid;
                } else {
                    hi = mid;
                }
            }
        }
        keys.push(good);
        a = good;
    }
    keys
}

#[derive(Clone, Copy, Debug)]
enum Method {
    Quant,
    Tdtr,
    GreedyLin,
    Hermite,
}

fn ship(tr: &Traj, m: Method, eps: f64, s: f64) -> Shipped {
    let n = tr.x.len();
    let keys: Vec<usize> = match m {
        Method::Quant => (0..n).collect(),
        Method::Tdtr => tdtr(tr, eps),
        Method::GreedyLin => greedy(n, eps, |a, b| lin_err(tr, a, b, s)),
        Method::Hermite => greedy(n, eps, |a, b| fit_hermite(tr, a, b, s).1),
    };
    let q = keys.iter().map(|&i| qpos(tr, i, s)).collect();
    let tan = matches!(m, Method::Hermite).then(|| {
        keys.windows(2)
            .map(|w| fit_hermite(tr, w[0], w[1], s).0)
            .collect()
    });
    Shipped { keys, q, tan }
}

fn run(trajs: &[Traj], m: Method, eps: f64, s: f64, unit: i64) -> (usize, usize, Err) {
    let chunks: Vec<(usize, usize, Err)> = trajs
        .par_chunks(500)
        .map(|chunk| {
            let mut buf = Vec::new();
            let mut err = Err::default();
            let mut kept = 0;
            for tr in chunk {
                let sh = ship(tr, m, eps, s);
                measure(tr, &sh, s, &mut err);
                kept += sh.keys.len();
                let tq: Vec<i64> = sh.keys.iter().map(|&i| tr.t_ms[i] / unit).collect();
                code_series(&sh.q, &tq, &mut buf);
                if let Some(tn) = &sh.tan {
                    // Tangents as residuals against the segment chord (a
                    // straight segment costs ~0), then plain varint.
                    let mut flat: Vec<[i64; 2]> = Vec::with_capacity(tn.len() * 2);
                    for (k, p) in tn.iter().enumerate() {
                        let c = [sh.q[k + 1][0] - sh.q[k][0], sh.q[k + 1][1] - sh.q[k][1]];
                        flat.push([p[0][0] - c[0], p[0][1] - c[1]]);
                        flat.push([p[1][0] - c[0], p[1][1] - c[1]]);
                    }
                    for v in flat {
                        put(&mut buf, zigzag(v[0]));
                        put(&mut buf, zigzag(v[1]));
                    }
                }
            }
            (zstd::bulk::compress(&buf, 3).unwrap().len(), kept, err)
        })
        .collect();
    chunks
        .into_iter()
        .fold((0, 0, Err::default()), |(b, k, e), (b2, k2, e2)| {
            (b + b2, k + k2, e.merge(e2))
        })
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: trajectory-lossy <traj.bin>");
    let trajs = load(&path);
    let nv: usize = trajs.iter().map(|t| t.x.len()).sum();
    let unit = time_unit_ms(&trajs);
    println!(
        "=== {}  {} trajectories, {} vertices, time unit {} ms",
        path,
        trajs.len(),
        nv,
        unit
    );
    println!(
        "{:<10} {:>8} {:>8} {:>9} {:>8} {:>9} {:>9} {:>9}",
        "method", "eps_m", "grid_m", "kept%", "B/vtx", "max_m", "p99_m", "mean_m"
    );
    // Lossless-at-1m reference, then budgets spanning z12 (~8 m) .. z4 (~2 km) quarter-pixels.
    for &budget in &[
        1.0f64, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 2048.0,
    ] {
        // Quantization-only must spend its whole budget on the grid: max error s/sqrt(2).
        let mut rows = vec![(Method::Quant, 0.0, budget * std::f64::consts::SQRT_2)];
        for split in [0.5f64, 0.8] {
            let g = (budget * (1.0 - split) * std::f64::consts::SQRT_2).max(0.5);
            rows.push((Method::Tdtr, budget * split, g));
            rows.push((Method::GreedyLin, budget, g));
            rows.push((Method::Hermite, budget, g));
        }
        for (m, eps, s) in rows {
            let (bytes, kept, err) = run(&trajs, m, eps, s, unit);
            println!(
                "{:<10} {:>8.1} {:>8.2} {:>8.2}% {:>8.3} {:>9.1} {:>9.1} {:>9.2}",
                format!("{:?}", m),
                budget,
                s,
                100.0 * kept as f64 / nv as f64,
                bytes as f64 / nv as f64,
                err.max,
                err.p99(),
                err.sum / err.n as f64
            );
        }
        println!();
    }
}
