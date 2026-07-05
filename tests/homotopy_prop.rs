//! Property test for the polyhedral homotopy driver: random complex
//! coefficients on the fixed mixed-volume-2 trinomial supports
//! `A₁ = {1, x, xy}`, `A₂ = {1, y, xy}`.
//!
//! The supports (and the seed) are fixed, so the offline half — lifting,
//! cells, mixed volume — is identical for every case; what varies is the
//! coefficient vector the online tracker actually has to follow. Magnitudes
//! are bounded away from zero so no edge coefficient degenerates a binomial
//! start system, but phases and magnitudes are otherwise arbitrary.
//!
//! Non-converged paths are tolerated per-case (a draw can sit close to the
//! discriminant), but every path that *claims* a root — plain converged or
//! a Cauchy-endgame singular endpoint — must deliver a genuine one, and a
//! systematically high failure rate would trip the counted assertion below.

use proptest::prelude::*;

use talrost::complex::c64;
use talrost::mvpoly::{MPoly, MSystem, Monomial};
use talrost::solvers::homotopy::{solve, PathStatus, TrackOptions};

/// A complex coefficient with magnitude in `[0.25, 4]` and arbitrary phase:
/// moderate (all arithmetic well in range) and bounded away from `0`.
fn coeff() -> impl Strategy<Value = c64> {
    (0.25..4.0f64, 0.0..core::f64::consts::TAU).prop_map(|(r, theta)| c64::from_polar(r, theta))
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    #[test]
    fn trinomial_solve_finds_verified_roots(
        a in prop::array::uniform3(coeff()),
        b in prop::array::uniform3(coeff()),
    ) {
        let f1 = MPoly::new(
            a,
            [
                Monomial::new([0, 0]),
                Monomial::new([1, 0]),
                Monomial::new([1, 1]),
            ],
        );
        let f2 = MPoly::new(
            b,
            [
                Monomial::new([0, 0]),
                Monomial::new([0, 1]),
                Monomial::new([1, 1]),
            ],
        );
        let system = MSystem::new([f1, f2]);

        // Fixed supports + fixed seed: the lifting is deterministic and
        // known-generic, so solve() must not error.
        let report = solve(&system, 2026, &TrackOptions::default()).unwrap();
        prop_assert_eq!(report.mixed_volume, 2);
        prop_assert_eq!(report.paths.len(), 2);

        let mut converged = 0usize;
        for p in &report.paths {
            if p.status == PathStatus::Converged {
                converged += 1;
                let h = system.eval(&p.point);
                let res = h[0].magnitude().max(h[1].magnitude());
                prop_assert!(
                    res < 1e-6,
                    "converged path has residual {} at {:?}",
                    res,
                    p.point
                );
                prop_assert!(p.pivot_ratio > 0.0);
            } else if p.status.is_root() {
                // A draw sitting close enough to the discriminant can end
                // at a (near-)singular root through the Cauchy endgame;
                // its endpoint must still satisfy the endgame's residual
                // gate against the target.
                let h = system.eval(&p.point);
                let res = h[0].magnitude().max(h[1].magnitude());
                let scale = p.point[0].magnitude().max(p.point[1].magnitude()).max(1.0);
                prop_assert!(
                    res < 1e-5 * scale,
                    "singular endpoint has residual {} at {:?}",
                    res,
                    p.point
                );
            }
        }
        prop_assert_eq!(converged, report.converged_count());
        prop_assert_eq!(
            report.failed_paths().count(),
            report.paths.len() - converged - report.singular_count()
        );
        // Coefficients bounded away from zero on these supports keep both
        // torus roots at moderate scale; wholesale failure of a draw would
        // flag a tracker bug rather than a degenerate instance.
        prop_assert!(
            converged >= 1,
            "no path converged for {:?} / {:?}",
            a,
            b
        );
    }
}
