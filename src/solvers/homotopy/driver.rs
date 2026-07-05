//! The end-to-end polyhedral homotopy driver: [`solve`] runs the whole
//! offline + online pipeline on one target system and reports every path.
//!
//! Pipeline (per Huber–Sturmfels): extract supports
//! ([`Support::from_msystem`]) → lift generically ([`random_liftings`]) →
//! enumerate fine mixed cells ([`mixed_cells`]) → solve each cell's binomial
//! start system ([`start_solutions`]) → track every start root to `t = 1`
//! along the cell homotopy ([`CellHomotopy`] + [`track_path`]). The number
//! of paths equals the mixed volume — Bernstein's generic root count on the
//! torus `(ℂ*)ⁿ` — and each converged endpoint approximates one root of the
//! target.

use crate::complex::Complex;
use crate::mvpoly::MSystem;
use crate::real::Real;

use super::cells::{mixed_cells, GenericityError, MixedCell};
use super::start::start_solutions;
use super::support::{random_liftings, Lifting, Support};
use super::track::{track_path, CellHomotopy, PathResult, PathStatus, TrackOptions};

/// The ∞-distance between two complex points, for solution dedup.
fn dist_inf<F: Real, const NV: usize>(a: &[Complex<F>; NV], b: &[Complex<F>; NV]) -> F {
    let mut m = F::ZERO;
    for (x, y) in a.iter().zip(b.iter()) {
        m = m.max((*x - *y).magnitude());
    }
    m
}

/// Everything [`solve`] found: one [`PathResult`] per tracked path (kept
/// raw and complete — failures included) plus the mixed volume they were
/// counted against. Deduplicated roots are a *view*
/// ([`SolveReport::distinct_solutions`]), never a mutation of the paths.
#[derive(Clone, Debug, PartialEq)]
pub struct SolveReport<F: Real, const NV: usize> {
    /// One result per tracked path, in deterministic (cell, start) order.
    pub paths: Vec<PathResult<F, NV>>,
    /// The mixed volume of the system's supports — the number of paths and
    /// the generic torus root count.
    pub mixed_volume: u64,
}

impl<F: Real, const NV: usize> SolveReport<F, NV> {
    /// The endpoints of the converged paths — the approximate roots of the
    /// target system, with multiplicity: near-multiple roots appear once
    /// per path that reached them.
    pub fn solutions(&self) -> impl Iterator<Item = [Complex<F>; NV]> + '_ {
        self.paths
            .iter()
            .filter(|p| p.status == PathStatus::Converged)
            .map(|p| p.point)
    }

    /// The converged endpoints with near-duplicates collapsed: a solution
    /// is kept when its pairwise ∞-distance to every already-kept solution
    /// exceeds `tol` (first occurrence wins, in path order). A view — the
    /// raw [`SolveReport::paths`] stay intact.
    pub fn distinct_solutions(&self, tol: F) -> Vec<[Complex<F>; NV]> {
        let mut out: Vec<[Complex<F>; NV]> = Vec::new();
        for s in self.solutions() {
            if !out.iter().any(|r| dist_inf(r, &s) <= tol) {
                out.push(s);
            }
        }
        out
    }
}

/// Lift with `seed` and enumerate the cells, as one retryable unit.
#[allow(clippy::type_complexity)]
fn lift_and_enumerate<F: Real, const NV: usize>(
    supports: &[Support<NV>; NV],
    seed: u64,
) -> Result<([Lifting<F>; NV], Vec<MixedCell<F, NV>>), GenericityError> {
    let liftings = random_liftings::<F, NV>(supports, seed);
    let cells = mixed_cells(supports, &liftings)?;
    Ok((liftings, cells))
}

/// Solves the square sparse system `F = 0` on the torus `(ℂ*)^NV` by
/// polyhedral homotopy continuation: one tracked path per unit of mixed
/// volume, starting from the binomial roots of the `seed`-lifted mixed
/// cells. `F` (the real scalar) is carried by the system argument, so no
/// turbofish is needed at the call site.
///
/// Genericity is checked, not assumed: if the seed's lifting produces a
/// near-tie in the cell enumeration, `solve` **re-lifts once** with
/// `seed + 1` before surfacing the [`GenericityError`] — one deterministic
/// retry keeps same-seed runs reproducible while absorbing the measure-zero
/// unlucky draw.
///
/// The report keeps every path, converged or not; use
/// [`SolveReport::solutions`] / [`SolveReport::distinct_solutions`] for the
/// root list. Paths whose true endpoint lies outside the torus or at
/// infinity end non-[`PathStatus::Converged`] (no endgames are
/// implemented), so a deficient system yields fewer solutions than
/// `mixed_volume` — honestly reported, never guessed.
pub fn solve<F: Real, const NV: usize, const MAXT: usize>(
    system: &MSystem<Complex<F>, NV, NV, MAXT>,
    seed: u64,
    options: &TrackOptions<F>,
) -> Result<SolveReport<F, NV>, GenericityError> {
    let supports = Support::from_msystem(system);
    let (liftings, cells) = match lift_and_enumerate::<F, NV>(&supports, seed) {
        Ok(ok) => ok,
        // One internal re-lift on a degenerate draw, then surface the error.
        Err(_) => lift_and_enumerate::<F, NV>(&supports, seed.wrapping_add(1))?,
    };

    let mut paths = Vec::new();
    let mut mixed_volume = 0u64;
    for cell in &cells {
        mixed_volume += cell.volume();
        let homotopy = CellHomotopy::new(system, &supports, &liftings, cell);
        for start in start_solutions(system, cell) {
            paths.push(track_path(&homotopy, start, options));
        }
    }
    Ok(SolveReport {
        paths,
        mixed_volume,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::complex::c64;
    use crate::matrix::Matrix;
    use crate::mvpoly::{MPoly, Monomial};
    use crate::polynomial::Polynomial;
    use crate::vector::Vector;

    fn residual_inf<const MAXT: usize>(sys: &MSystem<c64, 2, 2, MAXT>, x: &[c64; 2]) -> f64 {
        let h = sys.eval(x);
        h[0].magnitude().max(h[1].magnitude())
    }

    /// Validation 1: two generic complex linear equations (mixed volume 1).
    /// One path, converged, endpoint identical (to 1e-10) with the direct
    /// LU solve of the same linear system.
    #[test]
    fn linear_pair_matches_direct_lu_solve() {
        let c = c64::new;
        let monos = [
            Monomial::new([0, 0]),
            Monomial::new([1, 0]),
            Monomial::new([0, 1]),
        ];
        let f1 = MPoly::new([c(1.3, 0.4), c(-2.1, 0.9), c(0.7, -1.6)], monos);
        let f2 = MPoly::new([c(-0.5, -1.1), c(0.8, 0.3), c(1.9, 0.7)], monos);
        let system = MSystem::new([f1, f2]);

        let report = solve(&system, 1, &TrackOptions::default()).unwrap();
        assert_eq!(report.mixed_volume, 1);
        assert_eq!(report.paths.len(), 1);
        assert_eq!(report.paths[0].status, PathStatus::Converged);

        // Direct solve of b·x + c·y = −a per equation.
        let m = Matrix::new([[f1.coeffs[1], f1.coeffs[2]], [f2.coeffs[1], f2.coeffs[2]]]);
        let direct = m
            .solve(&Vector::new([-f1.coeffs[0], -f2.coeffs[0]]))
            .unwrap();
        let p = report.paths[0].point;
        assert!((p[0] - direct.b[0]).magnitude() < 1e-10);
        assert!((p[1] - direct.b[1]).magnitude() < 1e-10);
        assert_eq!(report.distinct_solutions(1e-6).len(), 1);
    }

    /// Validation 2: {x² = 2, y² = 3} through the whole pipeline. Mixed
    /// volume 4, all four paths converged, endpoints (±√2, ±√3).
    #[test]
    fn binomial_pair_through_the_pipeline() {
        let c = c64::new;
        let f1 = MPoly::new(
            [c(-2.0, 0.0), c(1.0, 0.0)],
            [Monomial::new([0, 0]), Monomial::new([2, 0])],
        );
        let f2 = MPoly::new(
            [c(-3.0, 0.0), c(1.0, 0.0)],
            [Monomial::new([0, 0]), Monomial::new([0, 2])],
        );
        let system = MSystem::new([f1, f2]);

        let report = solve(&system, 7, &TrackOptions::default()).unwrap();
        assert_eq!(report.mixed_volume, 4);
        assert_eq!(report.paths.len(), 4);
        assert!(report
            .paths
            .iter()
            .all(|p| p.status == PathStatus::Converged));

        let (r2, r3) = (2.0f64.sqrt(), 3.0f64.sqrt());
        for sx in [r2, -r2] {
            for sy in [r3, -r3] {
                assert!(
                    report
                        .solutions()
                        .any(|s| (s[0] - c(sx, 0.0)).magnitude() < 1e-8
                            && (s[1] - c(sy, 0.0)).magnitude() < 1e-8),
                    "root ({}, {}) not found",
                    sx,
                    sy
                );
            }
        }
        assert_eq!(report.distinct_solutions(1e-6).len(), 4);
    }

    /// Validation 3: the sparse trinomial pair with real coefficients,
    /// cross-checked against the eliminated univariate quadratic.
    ///
    /// With f = a + b·x + c·xy and g = d + e·y + f·xy, solving g for y
    /// (y = −d/(e + f·x), valid off the line e + f·x = 0) and substituting
    /// into f·(e + f·x) eliminates y:
    ///
    ///   (a + b·x)(e + f·x) − c·d·x
    ///     = b·f·x² + (a·f + b·e − c·d)·x + a·e = 0,
    ///
    /// the spec's quadratic up to an overall sign. The coefficients
    /// (a, b, c, d, e, f) = (1, −3, 1, 2, 1, 1) give −3x² − 4x + 1 (after
    /// that sign flip) with discriminant (−4)² − 4·(−3)·1 = 28 > 0: two
    /// real torus roots, verified below and recovered by the tracker.
    #[test]
    fn trinomial_pair_matches_eliminated_quadratic() {
        let (a, b, cc, d, e, f) = (1.0f64, -3.0, 1.0, 2.0, 1.0, 1.0);

        // The eliminated quadratic in ascending coefficient order
        // [a·e, a·f + b·e − c·d, b·f], solved by the existing univariate
        // machinery (Blinn's quadratic).
        let quad = Polynomial::new([a * e, a * f + b * e - cc * d, b * f]);
        let disc = (a * f + b * e - cc * d) * (a * f + b * e - cc * d) - 4.0 * (b * f) * (a * e);
        assert!(disc > 0.0, "coefficients must give two real roots");
        let roots = quad.roots(1e-12);
        assert_eq!(roots.len(), 2);

        let z = |v: f64| c64::new(v, 0.0);
        let f1 = MPoly::new(
            [z(a), z(b), z(cc)],
            [
                Monomial::new([0, 0]),
                Monomial::new([1, 0]),
                Monomial::new([1, 1]),
            ],
        );
        let f2 = MPoly::new(
            [z(d), z(e), z(f)],
            [
                Monomial::new([0, 0]),
                Monomial::new([0, 1]),
                Monomial::new([1, 1]),
            ],
        );
        let system = MSystem::new([f1, f2]);

        let report = solve(&system, 3, &TrackOptions::default()).unwrap();
        assert_eq!(report.mixed_volume, 2);
        assert_eq!(report.paths.len(), 2);
        assert!(report
            .paths
            .iter()
            .all(|p| p.status == PathStatus::Converged));

        // Every quadratic root is some endpoint's x-coordinate.
        for r in &roots {
            assert!(
                report
                    .solutions()
                    .any(|s| (s[0] - z(*r)).magnitude() < 1e-8),
                "quadratic root {} not among the endpoints",
                r
            );
        }
        // And every endpoint is a genuine root of the target.
        for p in &report.paths {
            assert!(residual_inf(&system, &p.point) < 1e-8);
        }
    }

    /// Validation 4: a dense generic conic pair (fixed complex
    /// coefficients). Mixed volume 4, four converged paths, residuals below
    /// 1e-8, endpoints pairwise distinct.
    #[test]
    fn dense_conic_pair() {
        let c = c64::new;
        let monos = [
            Monomial::new([0, 0]),
            Monomial::new([1, 0]),
            Monomial::new([0, 1]),
            Monomial::new([2, 0]),
            Monomial::new([1, 1]),
            Monomial::new([0, 2]),
        ];
        let f1 = MPoly::new(
            [
                c(1.1, 0.3),
                c(-0.7, 0.9),
                c(0.5, -1.3),
                c(2.0, 0.1),
                c(-1.4, -0.8),
                c(0.6, 1.7),
            ],
            monos,
        );
        let f2 = MPoly::new(
            [
                c(-0.9, 1.2),
                c(1.8, -0.4),
                c(0.3, 0.7),
                c(-1.1, -1.6),
                c(0.8, 0.2),
                c(1.5, -0.5),
            ],
            monos,
        );
        let system = MSystem::new([f1, f2]);

        let report = solve(&system, 4, &TrackOptions::default()).unwrap();
        assert_eq!(report.mixed_volume, 4);
        assert_eq!(report.paths.len(), 4);
        assert!(report
            .paths
            .iter()
            .all(|p| p.status == PathStatus::Converged));
        for p in &report.paths {
            assert!(residual_inf(&system, &p.point) < 1e-8);
        }
        // No oracle needed: residuals + the count of distinct endpoints.
        assert_eq!(report.distinct_solutions(1e-6).len(), 4);
    }

    /// Validation 5: failure honesty. Two proportional equations are
    /// rank-deficient at every point of the target (t = 1), so no path can
    /// converge — but nothing panics and the report is returned in full.
    #[test]
    fn proportional_equations_fail_honestly() {
        let z = |v: f64| c64::new(v, 0.0);
        let monos = [
            Monomial::new([0, 0]),
            Monomial::new([1, 0]),
            Monomial::new([0, 1]),
        ];
        let f1 = MPoly::new([z(1.0), z(1.0), z(1.0)], monos);
        let f2 = MPoly::new([z(2.0), z(2.0), z(2.0)], monos); // g = 2·f
        let system = MSystem::new([f1, f2]);

        let report = solve(&system, 5, &TrackOptions::default()).unwrap();
        assert_eq!(report.mixed_volume, 1);
        assert_eq!(report.paths.len(), 1);
        assert!(
            report
                .paths
                .iter()
                .all(|p| p.status != PathStatus::Converged),
            "a rank-deficient target must not report convergence, got {:?}",
            report.paths[0].status
        );
        assert!(report.distinct_solutions(1e-6).is_empty());
        // The failed path still reports where it got to.
        assert!(report.paths[0].t_reached < 1.0);
    }

    /// Validation 6: determinism — the same seed gives a bit-for-bit
    /// identical report (statuses, endpoints, step counts).
    #[test]
    fn same_seed_same_report() {
        let c = c64::new;
        let f1 = MPoly::new(
            [c(0.83, -1.21), c(-0.44, 0.67), c(1.92, 0.35)],
            [
                Monomial::new([0, 0]),
                Monomial::new([1, 0]),
                Monomial::new([1, 1]),
            ],
        );
        let f2 = MPoly::new(
            [c(-1.37, 0.58), c(0.29, 1.74), c(-0.91, -0.62)],
            [
                Monomial::new([0, 0]),
                Monomial::new([0, 1]),
                Monomial::new([1, 1]),
            ],
        );
        let system = MSystem::new([f1, f2]);

        let a = solve(&system, 42, &TrackOptions::default()).unwrap();
        let b = solve(&system, 42, &TrackOptions::default()).unwrap();
        assert_eq!(a, b); // PathResult is PartialEq: bit-for-bit endpoints

        // A different seed lifts differently but finds the same torus roots.
        let other = solve(&system, 43, &TrackOptions::default()).unwrap();
        assert_eq!(other.mixed_volume, a.mixed_volume);
        let sa = a.distinct_solutions(1e-6);
        let so = other.distinct_solutions(1e-6);
        assert_eq!(sa.len(), so.len());
        for s in &sa {
            assert!(so.iter().any(|r| dist_inf(r, s) < 1e-6));
        }
    }
}
