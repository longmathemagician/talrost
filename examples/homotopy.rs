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

    let report = solve(&system, 2026, &TrackOptions::default()).expect("generic lifting");
    println!(
        "\nmixed volume (Bernstein torus root count): {}",
        report.mixed_volume
    );

    println!("\npaths:");
    for (i, path) in report.paths.iter().enumerate() {
        println!(
            "  #{}: {:?} at t = {} ({} steps, {} Newton iterations)",
            i, path.status, path.t_reached, path.steps, path.newton_iters
        );
    }

    let solutions = report.distinct_solutions(1e-6);
    println!("\ndistinct solutions ({}):", solutions.len());
    for s in &solutions {
        let residual = system.eval(s);
        let res_inf = residual[0].magnitude().max(residual[1].magnitude());
        println!("  x = {}, y = {}   with ‖F‖∞ = {:.3e}", s[0], s[1], res_inf);
        assert!(res_inf < 1e-8, "endpoint failed its residual check");
    }
    assert_eq!(solutions.len(), 2);
    println!("\nall residuals below 1e-8 — every endpoint is a verified root");
}
