//! Micro-benchmarks with no dependencies beyond `std::time` and
//! `std::hint::black_box`.
//!
//! Run with:
//!
//! ```text
//! cargo run --release --example bench
//! cargo +nightly run --release --features specialization --example bench
//! ```
//!
//! The second form swaps the naive FMA matmul loop for the
//! multiplication-saving kernels (Strassen 2×2, Laderman 3×3,
//! AlphaTensor-style 4×4) so the two can be compared. Numbers are wall-clock
//! nanoseconds per operation, medians of several repetitions; treat them as
//! order-of-magnitude indicators, not rigorous statistics.

use std::hint::black_box;
use std::time::Instant;

use talrost::complex::c64;
use talrost::matrix::Matrix;
use talrost::mvpoly::{MPoly, MSystem, Monomial};
use talrost::polynomial::Polynomial;
use talrost::solvers::homotopy::{
    mixed_cells, random_liftings, solve, start_solutions, CellHomotopy, Support, TrackOptions,
};
use talrost::vector::Vector;

/// Times `f` over `iters` iterations, repeated `REPS` times; returns the
/// median nanoseconds per iteration.
fn time_ns_per_op<F: FnMut()>(iters: u64, mut f: F) -> f64 {
    const REPS: usize = 7;
    let mut samples = [0.0f64; REPS];
    for s in samples.iter_mut() {
        let start = Instant::now();
        for _ in 0..iters {
            f();
        }
        *s = start.elapsed().as_nanos() as f64 / iters as f64;
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[REPS / 2]
}

fn main() {
    let mut rows: Vec<(&str, f64)> = Vec::new();

    // Horner evaluation of an f64 quartic (5 coefficients, ascending).
    let p = Polynomial::new([4.0, -3.0, 2.0, -1.0, 0.5]);
    rows.push((
        "Polynomial<f64,5>::eval (Horner quartic)",
        time_ns_per_op(10_000_000, || {
            black_box(black_box(&p).eval(black_box(1.7)));
        }),
    ));

    // The same real quartic evaluated at a complex point through eval_at.
    let z = c64::new(0.8, -0.6);
    rows.push((
        "Polynomial<f64,5>::eval_at::<c64>",
        time_ns_per_op(10_000_000, || {
            black_box(black_box(&p).eval_at(black_box(z)));
        }),
    ));

    // 4×4 f64 matrix multiplication. Default builds run the naive FMA loop;
    // `--features specialization` (nightly) dispatches the AlphaTensor-style
    // 4×4 kernel instead.
    let a = Matrix::<f64, 4, 4>::new([
        [1.0, -2.0, 3.0, -4.0],
        [5.0, 6.0, -7.0, 8.0],
        [-9.0, 10.0, 11.0, -12.0],
        [13.0, -14.0, 15.0, 16.0],
    ]);
    let b = Matrix::<f64, 4, 4>::new([
        [-16.0, 15.0, -14.0, 13.0],
        [12.0, -11.0, 10.0, -9.0],
        [8.0, 7.0, -6.0, 5.0],
        [-4.0, 3.0, 2.0, -1.0],
    ]);
    let kernel = if cfg!(feature = "specialization") {
        "matmul 4x4 f64 (specialization kernels)"
    } else {
        "matmul 4x4 f64 (naive loop)"
    };
    rows.push((
        kernel,
        time_ns_per_op(5_000_000, || {
            black_box(black_box(a) * black_box(b));
        }),
    ));

    // A mock Newton-corrector step for the 2×2 system
    //   f1 = x² + y² − 5,  f2 = xy − 2      (root at (2, 1)):
    // one eval_jacobian sweep (DualN forward AD) plus one LU solve.
    let f1 = MPoly::<f64, 2, 3>::new(
        [1.0, 1.0, -5.0],
        [
            Monomial::new([2, 0]),
            Monomial::new([0, 2]),
            Monomial::new([0, 0]),
        ],
    );
    let f2 = MPoly::<f64, 2, 3>::new(
        [1.0, -2.0, 0.0],
        [
            Monomial::new([1, 1]),
            Monomial::new([0, 0]),
            Monomial::new([0, 0]),
        ],
    );
    let sys = MSystem::new([f1, f2]);
    rows.push((
        "corrector step (eval_jacobian + solve, 2x2)",
        time_ns_per_op(2_000_000, || {
            let x = black_box([2.1, 0.9]);
            let (h, j) = sys.eval_jacobian(&x);
            black_box(j.solve(&Vector::new([-h[0], -h[1]])));
        }),
    ));

    // One polyhedral-homotopy tracker step, mid-path on a dense conic pair:
    // Euler predict (tangent solve) plus three Newton corrections, complex
    // 2×2 all the way down.
    let conic_monos = [
        Monomial::new([0, 0]),
        Monomial::new([1, 0]),
        Monomial::new([0, 1]),
        Monomial::new([2, 0]),
        Monomial::new([1, 1]),
        Monomial::new([0, 2]),
    ];
    let g1 = MPoly::new(
        [
            c64::new(1.1, 0.3),
            c64::new(-0.7, 0.9),
            c64::new(0.5, -1.3),
            c64::new(2.0, 0.1),
            c64::new(-1.4, -0.8),
            c64::new(0.6, 1.7),
        ],
        conic_monos,
    );
    let g2 = MPoly::new(
        [
            c64::new(-0.9, 1.2),
            c64::new(1.8, -0.4),
            c64::new(0.3, 0.7),
            c64::new(-1.1, -1.6),
            c64::new(0.8, 0.2),
            c64::new(1.5, -0.5),
        ],
        conic_monos,
    );
    let conic = MSystem::new([g1, g2]);
    let supports = Support::from_msystem(&conic);
    let liftings = random_liftings::<f64, 2>(&supports, 4);
    let cells = mixed_cells(&supports, &liftings).expect("generic lifting");
    let hom = CellHomotopy::new(&conic, &supports, &liftings, &cells[0]);
    // Walk a path to t = 0.5 by corrector-only continuation so the benched
    // step runs from a genuine mid-path point.
    let mut y = start_solutions(&conic, &cells[0])[0];
    for k in 1..=20 {
        let t = k as f64 * 0.025;
        for _ in 0..3 {
            let (h, j) = hom.eval_jacobian(&y, t);
            let d = j.solve(&Vector::new([-h[0], -h[1]])).unwrap();
            y = [y[0] + d.b[0], y[1] + d.b[1]];
        }
    }
    let tracker_step = |y0: &[c64; 2], t: f64, dt: f64| -> [c64; 2] {
        // Predict: J·ẏ = −H_t, Euler.
        let (_, j) = hom.eval_jacobian(y0, t);
        let ht = hom.dt(y0, t);
        let v = j.lu().unwrap().solve(&Vector::new([-ht[0], -ht[1]]));
        let mut yt = [y0[0] + v.b[0] * dt, y0[1] + v.b[1] * dt];
        // Correct: three Newton iterations at fixed t + dt, with the same
        // update-norm test the tracker performs.
        for _ in 0..3 {
            let (h, j) = hom.eval_jacobian(&yt, t + dt);
            let d = j.lu().unwrap().solve(&Vector::new([-h[0], -h[1]]));
            yt = [yt[0] + d.b[0], yt[1] + d.b[1]];
            let norm = d.b[0].magnitude().max(d.b[1].magnitude());
            if norm <= 1e-10 * yt[0].magnitude().max(yt[1].magnitude()).max(1.0) {
                break;
            }
        }
        yt
    };
    rows.push((
        "homotopy tracker step (conic pair, mid-path)",
        time_ns_per_op(200_000, || {
            black_box(tracker_step(black_box(&y), black_box(0.5), black_box(0.05)));
        }),
    ));

    // A full polyhedral solve() of the sparse trinomial pair (mixed
    // volume 2): offline lift/cells/starts plus two tracked paths.
    let t1 = MPoly::new(
        [c64::new(1.0, 0.0), c64::new(-3.0, 0.0), c64::new(1.0, 0.0)],
        [
            Monomial::new([0, 0]),
            Monomial::new([1, 0]),
            Monomial::new([1, 1]),
        ],
    );
    let t2 = MPoly::new(
        [c64::new(2.0, 0.0), c64::new(1.0, 0.0), c64::new(1.0, 0.0)],
        [
            Monomial::new([0, 0]),
            Monomial::new([0, 1]),
            Monomial::new([1, 1]),
        ],
    );
    let trinomial = MSystem::new([t1, t2]);
    rows.push((
        "homotopy solve() (trinomial pair, MV 2)",
        time_ns_per_op(1_000, || {
            black_box(
                solve(
                    black_box(&trinomial),
                    black_box(2026),
                    &TrackOptions::default(),
                )
                .unwrap(),
            );
        }),
    ));

    println!(
        "talrost micro-benchmarks ({} build, specialization: {})",
        if cfg!(debug_assertions) {
            "debug -- numbers are meaningless, use --release"
        } else {
            "release"
        },
        cfg!(feature = "specialization"),
    );
    println!("{:-<62}", "");
    println!("{:<48} {:>12}", "benchmark", "ns/op");
    println!("{:-<62}", "");
    for (name, ns) in rows {
        println!("{:<48} {:>12.2}", name, ns);
    }
}
