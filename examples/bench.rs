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
