//! End-to-end polyhedral homotopy continuation: solve a sparse polynomial
//! system and verify every root, exercising the whole stack — supports,
//! generic lifting, mixed cells, binomial start systems, and path tracking.
//!
//! Run with `cargo run --example homotopy`.

use talrost::complex::c64;
use talrost::mvpoly::{MPoly, MSystem, Monomial};
use talrost::solvers::homotopy::{solve, TrackOptions};

fn main() {
    // The sparse trinomial pair
    //   f(x, y) = 1 − 3x + xy
    //   g(x, y) = 2 +  y + xy
    // has Bézout bound 4 but mixed volume 2 — the polyhedral advantage:
    // only two paths are tracked, and eliminating y confirms exactly two
    // torus roots (the quadratic −3x² − 4x + 1 = 0 in x).
    let z = |v: f64| c64::new(v, 0.0);
    let f = MPoly::new(
        [z(1.0), z(-3.0), z(1.0)],
        [
            Monomial::new([0, 0]),
            Monomial::new([1, 0]),
            Monomial::new([1, 1]),
        ],
    );
    let g = MPoly::new(
        [z(2.0), z(1.0), z(1.0)],
        [
            Monomial::new([0, 0]),
            Monomial::new([0, 1]),
            Monomial::new([1, 1]),
        ],
    );
    let system = MSystem::new([f, g]);

    println!("target system:");
    print!("{}", system);

    // TrackOptions selects the predictor (Euler / Rk2 / Rk4); the default
    // is the benchmarked best (Rk4) — spelled out here only for show.
    let options = TrackOptions {
        predictor: talrost::solvers::homotopy::Predictor::Rk4,
        ..TrackOptions::default()
    };
    let report = solve(&system, 2026, &options).expect("generic lifting");

    // SolveReport implements Display: one summary line, then per-path
    // status, t reached, step/Newton counts, and the pivot-ratio
    // conditioning hint from the final polished Newton solve.
    println!();
    print!("{}", report);
    assert_eq!(report.converged_count(), 2);
    assert_eq!(report.failed_paths().count(), 0);

    let solutions = report.distinct_solutions(1e-6);
    println!("\ndistinct solutions ({}):", solutions.len());
    for s in &solutions {
        let residual = system.eval(s);
        let res_inf = residual[0].magnitude().max(residual[1].magnitude());
        println!("  x = {}, y = {}   with ‖F‖∞ = {:.3e}", s[0], s[1], res_inf);
        assert!(res_inf < 1e-8, "endpoint failed its residual check");
    }
    assert_eq!(solutions.len(), 2);

    // Both roots of this system happen to be real — real_solutions filters
    // (without zeroing the imaginary dust; the points come back as-is).
    let real = report.real_solutions(1e-8).count();
    println!("\nreal solutions (|im| <= 1e-8·max(1, |re|)): {}", real);
    assert_eq!(real, 2);
    println!("all residuals below 1e-8 — every endpoint is a verified root");
}
