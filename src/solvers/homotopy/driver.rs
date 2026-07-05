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
    /// The endpoints of the root-bearing paths — the approximate roots of
    /// the target system, with multiplicity: multiple roots appear once
    /// per path that reached them.
    ///
    /// **Includes singular endpoints**: paths that ended
    /// [`PathStatus::ConvergedSingular`] contribute their Cauchy-mean
    /// endpoints alongside the plain [`PathStatus::Converged`] ones — they
    /// passed the endgame's residual gate, so they are verified roots,
    /// just roots at which the Jacobian is singular. They stay *flagged*
    /// through [`SolveReport::paths`] (`status` carries the winding) and
    /// countable via [`SolveReport::singular_count`]; callers that want
    /// regular roots only can filter on the status themselves. Note the
    /// caveat on [`PathStatus::ConvergedSingular`]: for a target with a
    /// positive-dimensional solution set, these endpoints are genuine
    /// solution points but **not isolated roots**.
    pub fn solutions(&self) -> impl Iterator<Item = [Complex<F>; NV]> + '_ {
        self.paths
            .iter()
            .filter(|p| p.status.is_root())
            .map(|p| p.point)
    }

    /// The [`SolveReport::solutions`] endpoints (singular ones included)
    /// with near-duplicates collapsed: a solution is kept when its
    /// pairwise ∞-distance to every already-kept solution exceeds `tol`
    /// (first occurrence wins, in path order) — so a `w`-fold root appears
    /// once here even though `w` paths land on it. A view — the raw
    /// [`SolveReport::paths`] stay intact.
    pub fn distinct_solutions(&self, tol: F) -> Vec<[Complex<F>; NV]> {
        let mut out: Vec<[Complex<F>; NV]> = Vec::new();
        for s in self.solutions() {
            if !out.iter().any(|r| dist_inf(r, &s) <= tol) {
                out.push(s);
            }
        }
        out
    }

    /// How many paths reached `t = 1` with status
    /// [`PathStatus::Converged`] — equal to `mixed_volume` when nothing went
    /// wrong, smaller for deficient systems (roots at infinity, outside the
    /// torus, singular endpoints — counted separately by
    /// [`SolveReport::singular_count`] — or tracking failures).
    pub fn converged_count(&self) -> usize {
        self.paths
            .iter()
            .filter(|p| p.status == PathStatus::Converged)
            .count()
    }

    /// How many paths ended [`PathStatus::ConvergedSingular`] — endpoints
    /// computed by the Cauchy endgame at roots with a singular Jacobian.
    /// Disjoint from [`SolveReport::converged_count`]; a fully regular
    /// solve has `singular_count() == 0`.
    pub fn singular_count(&self) -> usize {
        self.paths
            .iter()
            .filter(|p| matches!(p.status, PathStatus::ConvergedSingular { .. }))
            .count()
    }

    /// The number of paths (plain [`PathStatus::Converged`] or
    /// [`PathStatus::ConvergedSingular`]) whose endpoint lies within `tol`
    /// (∞-norm) of `point` — the **multiplicity** of that root as the
    /// homotopy sees it: a `w`-fold isolated root attracts exactly `w`
    /// paths, so `w` paths land there and (when they form one monodromy
    /// cycle) each reports `ConvergedSingular { winding: w }`. Returns `0`
    /// when no path landed near `point`.
    pub fn multiplicity_of(&self, point: &[Complex<F>; NV], tol: F) -> usize {
        self.paths
            .iter()
            .filter(|p| p.status.is_root() && dist_inf(&p.point, point) <= tol)
            .count()
    }

    /// The paths that did **not** end on a root (neither
    /// [`PathStatus::Converged`] nor [`PathStatus::ConvergedSingular`]),
    /// with their honest last state: where they stopped
    /// ([`PathResult::t_reached`]), why ([`PathResult::status`]), and the
    /// conditioning hint of their last accepted step
    /// ([`PathResult::pivot_ratio`]).
    pub fn failed_paths(&self) -> impl Iterator<Item = &PathResult<F, NV>> + '_ {
        self.paths.iter().filter(|p| !p.status.is_root())
    }

    /// The [`SolveReport::solutions`] endpoints (singular ones included)
    /// that are real to within `tol`: every
    /// coordinate satisfies `|im| ≤ tol · max(1, |re|)`. Points are
    /// returned **as-is**, imaginary dust included — the filter classifies,
    /// it does not zero anything out, because fabricating exact realness
    /// would erase precisely the residual information the tolerance
    /// judgment was made from. Like [`SolveReport::solutions`], multiple
    /// paths landing on one root yield repeated entries.
    pub fn real_solutions(&self, tol: F) -> impl Iterator<Item = [Complex<F>; NV]> + '_ {
        self.solutions()
            .filter(move |s| s.iter().all(|z| z.im.abs() <= tol * F::ONE.max(z.re.abs())))
    }
}

impl<F, const NV: usize> core::fmt::Display for SolveReport<F, NV>
where
    F: Real + core::fmt::Display + core::fmt::LowerExp,
{
    /// One summary line (with a singular-endpoint tally when the Cauchy
    /// endgame certified any), then one line per path — status, `t`
    /// reached, step and Newton-iteration counts, and the
    /// [`PathResult::pivot_ratio`] conditioning hint. Allocation-free
    /// (`write!` only). The extra `Display`/`LowerExp` bounds on `F` are
    /// satisfied by `f32`/`f64`.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "mixed volume {}: {}/{} paths converged",
            self.mixed_volume,
            self.converged_count(),
            self.paths.len()
        )?;
        if self.singular_count() > 0 {
            write!(f, " (+{} singular)", self.singular_count())?;
        }
        writeln!(f)?;
        for (i, p) in self.paths.iter().enumerate() {
            writeln!(
                f,
                "  path {:>3}: {:<13} t = {:<7.5}  steps = {:>4}  newton = {:>4}  pivot_ratio = {:.2e}",
                i, p.status, p.t_reached, p.steps, p.newton_iters, p.pivot_ratio
            )?;
        }
        Ok(())
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
/// root list. Paths ending at **singular** isolated roots are finished by
/// the Cauchy endgame ([`PathStatus::ConvergedSingular`] carries the
/// winding; [`SolveReport::multiplicity_of`] counts the paths on a root).
/// Paths whose true endpoint lies outside the torus or at infinity still
/// end non-root ([`PathStatus::Diverged`] and friends), so a deficient
/// system yields fewer solutions than `mixed_volume` — honestly reported,
/// never guessed.
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
        // The no-trigger proof: a dense regular system stays endgame-free.
        assert!(report.paths.iter().all(|p| !p.endgame_entered));
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

    /// Validation 7: **cyclic-3** — the first 3-variable end-to-end system,
    /// and the reason the γ-twist exists (see [`CellHomotopy`]): its real
    /// symmetric coefficients are maximally non-generic, and without the
    /// twist all six paths fold pairwise on the discriminant at one interior
    /// `t` (conjugate-pair collisions, pivot ratios → 1e-8) and die with
    /// `MinStepReached`.
    ///
    /// Structural oracle: x + y + z = 0, xy + yz + zx = 0, xyz = 1 say the
    /// coordinates of every root are the roots of λ³ − 0·λ² + 0·λ − 1 =
    /// λ³ − 1, i.e. each solution is a permutation of (1, ω, ω̄) with
    /// ω = e^{2πi/3} — six roots in all (= the mixed volume), every
    /// coordinate of modulus 1, coordinate-sum 0, coordinate-product 1.
    #[test]
    fn cyclic_3_all_six_roots() {
        let z = |v: f64| c64::new(v, 0.0);
        let f1 = MPoly::new(
            [z(1.0), z(1.0), z(1.0)],
            [
                Monomial::new([1, 0, 0]),
                Monomial::new([0, 1, 0]),
                Monomial::new([0, 0, 1]),
            ],
        );
        let f2 = MPoly::new(
            [z(1.0), z(1.0), z(1.0)],
            [
                Monomial::new([1, 1, 0]),
                Monomial::new([0, 1, 1]),
                Monomial::new([1, 0, 1]),
            ],
        );
        // xyz − 1, padded to MAXT = 3 with a zero-coefficient term.
        let f3 = MPoly::new(
            [z(-1.0), z(1.0), z(0.0)],
            [
                Monomial::new([0, 0, 0]),
                Monomial::new([1, 1, 1]),
                Monomial::new([0, 0, 0]),
            ],
        );
        let system = MSystem::new([f1, f2, f3]);

        let report = solve(&system, 1, &TrackOptions::default()).unwrap();
        assert_eq!(report.mixed_volume, 6);
        assert_eq!(report.paths.len(), 6);
        assert_eq!(report.converged_count(), 6);
        assert_eq!(report.singular_count(), 0);
        assert_eq!(report.failed_paths().count(), 0);
        assert!(report
            .paths
            .iter()
            .all(|p| p.status == PathStatus::Converged));
        // The no-trigger proof: every path converged the regular way.
        assert!(report.paths.iter().all(|p| !p.endgame_entered));

        // Residuals: every endpoint is a genuine root of the target.
        for p in &report.paths {
            let h = system.eval(&p.point);
            let res = h[0].magnitude().max(h[1].magnitude()).max(h[2].magnitude());
            assert!(res < 1e-8, "residual {} too large", res);
            // The roots are simple and well-scaled: the conditioning hint
            // from the final polish must be comfortably away from 0.
            assert!(p.pivot_ratio > 1e-3, "pivot_ratio {}", p.pivot_ratio);
        }

        // Six pairwise-distinct solutions.
        let sols = report.distinct_solutions(1e-6);
        assert_eq!(sols.len(), 6);

        // Structural oracle: each solution is a permutation of (1, ω, ω̄).
        let omega = c64::nth_root_of_unity(1, 3);
        let cube_roots = [c64::new(1.0, 0.0), omega, omega * omega];
        for s in &sols {
            let mut sum = c64::new(0.0, 0.0);
            let mut prod = c64::new(1.0, 0.0);
            for c in s {
                assert!((c.magnitude() - 1.0).abs() < 1e-8, "|coord| != 1: {}", c);
                sum += *c;
                prod *= *c;
            }
            assert!(sum.magnitude() < 1e-8, "coordinate-sum {} != 0", sum);
            assert!(
                (prod - c64::new(1.0, 0.0)).magnitude() < 1e-8,
                "coordinate-product {} != 1",
                prod
            );
            // Genuinely a permutation: every coordinate is one of the cube
            // roots of unity, and all three are distinct.
            for c in s {
                assert!(
                    cube_roots.iter().any(|r| (*c - *r).magnitude() < 1e-6),
                    "coordinate {} is not a cube root of unity",
                    c
                );
            }
            assert!((s[0] - s[1]).magnitude() > 1e-6);
            assert!((s[0] - s[2]).magnitude() > 1e-6);
            assert!((s[1] - s[2]).magnitude() > 1e-6);
        }
    }

    /// Validation 8: the trinomial pair end-to-end over `Complex<f32>` with
    /// tolerances loosened to single precision — the solver is genuinely
    /// generic over [`Real`], not `f64`-only.
    #[test]
    fn trinomial_pair_end_to_end_in_f32() {
        use crate::complex::c32;
        let z = |v: f32| c32::new(v, 0.0);
        let f1 = MPoly::new(
            [z(1.0), z(-3.0), z(1.0)],
            [
                Monomial::new([0, 0]),
                Monomial::new([1, 0]),
                Monomial::new([1, 1]),
            ],
        );
        let f2 = MPoly::new(
            [z(2.0), z(1.0), z(1.0)],
            [
                Monomial::new([0, 0]),
                Monomial::new([0, 1]),
                Monomial::new([1, 1]),
            ],
        );
        let system = MSystem::new([f1, f2]);

        let options = TrackOptions::<f32> {
            newton_tol: 1e-5,
            ..TrackOptions::default()
        };
        let report = solve(&system, 3, &options).unwrap();
        assert_eq!(report.mixed_volume, 2);
        assert_eq!(report.converged_count(), 2);
        assert!(report.paths.iter().all(|p| !p.endgame_entered));
        for p in &report.paths {
            let h = system.eval(&p.point);
            let res = h[0].magnitude().max(h[1].magnitude());
            assert!(res < 1e-4, "f32 residual {} too large", res);
        }
        assert_eq!(report.distinct_solutions(1e-3).len(), 2);
    }

    /// Validation 9: the corrector-informed step control does not regress
    /// the total step count on the dense conic pair vs the Phase 8 rule.
    /// The Phase 8 configuration (fixed grow-on-success, Euler predictor —
    /// the only one it had) measured **419 total steps** on this exact
    /// system and seed; the Phase 9 defaults must not exceed that.
    #[test]
    fn step_control_does_not_regress_on_the_conic() {
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
        assert_eq!(report.converged_count(), 4);
        let total_steps: u32 = report.paths.iter().map(|p| p.steps).sum();
        assert!(
            total_steps <= 419,
            "step-control regression: {} total steps vs the Phase 8 rule's 419",
            total_steps
        );
    }

    /// The report helpers and `Display` impls: counts add up, the real
    /// filter classifies without mutating, and the formatted report carries
    /// one summary line plus one line per path.
    #[test]
    fn report_helpers_and_display() {
        // The real trinomial pair: mixed volume 2, both roots real
        // (eliminant discriminant 28 > 0 — see validation 3).
        let z = |v: f64| c64::new(v, 0.0);
        let f1 = MPoly::new(
            [z(1.0), z(-3.0), z(1.0)],
            [
                Monomial::new([0, 0]),
                Monomial::new([1, 0]),
                Monomial::new([1, 1]),
            ],
        );
        let f2 = MPoly::new(
            [z(2.0), z(1.0), z(1.0)],
            [
                Monomial::new([0, 0]),
                Monomial::new([0, 1]),
                Monomial::new([1, 1]),
            ],
        );
        let system = MSystem::new([f1, f2]);
        let report = solve(&system, 3, &TrackOptions::default()).unwrap();

        assert_eq!(report.converged_count(), 2);
        assert_eq!(report.failed_paths().count(), 0);
        assert_eq!(
            report.converged_count() + report.failed_paths().count(),
            report.paths.len()
        );
        // Both roots are real; the filter returns them with their imaginary
        // dust intact (no coordinate is zeroed).
        let real: Vec<_> = report.real_solutions(1e-8).collect();
        assert_eq!(real.len(), 2);
        assert!(report
            .real_solutions(1e-8)
            .zip(report.solutions())
            .all(|(r, s)| r == s));
        // Zero tolerance excludes everything with any imaginary dust at
        // all, or keeps exact-real points — either way, a subset.
        assert!(report.real_solutions(0.0).count() <= 2);
        // Every converged path carries a positive conditioning hint.
        assert!(report.paths.iter().all(|p| p.pivot_ratio > 0.0));

        // Display: one summary line + one line per path, with the pieces.
        let text = format!("{}", report);
        assert_eq!(text.lines().count(), 1 + report.paths.len());
        assert!(text.starts_with("mixed volume 2: 2/2 paths converged"));
        assert!(text.contains("path   0: converged"));
        assert!(text.contains("pivot_ratio ="));

        // A system with no real roots: x² + 1 = 0, y² − 3 = 0 has roots
        // (±i, ±√3) — solutions() yields 4, real_solutions() none.
        let g1 = MPoly::new(
            [z(1.0), z(1.0)],
            [Monomial::new([0, 0]), Monomial::new([2, 0])],
        );
        let g2 = MPoly::new(
            [z(-3.0), z(1.0)],
            [Monomial::new([0, 0]), Monomial::new([0, 2])],
        );
        let no_real = solve(&MSystem::new([g1, g2]), 7, &TrackOptions::default()).unwrap();
        assert_eq!(no_real.converged_count(), 4);
        assert_eq!(no_real.real_solutions(1e-8).count(), 0);

        // A failing report formats too, with the failure status visible.
        let h1 = MPoly::new(
            [z(1.0), z(1.0), z(1.0)],
            [
                Monomial::new([0, 0]),
                Monomial::new([1, 0]),
                Monomial::new([0, 1]),
            ],
        );
        let h2 = MPoly::new(
            [z(2.0), z(2.0), z(2.0)],
            [
                Monomial::new([0, 0]),
                Monomial::new([1, 0]),
                Monomial::new([0, 1]),
            ],
        );
        let failing = solve(&MSystem::new([h1, h2]), 5, &TrackOptions::default()).unwrap();
        assert_eq!(failing.converged_count(), 0);
        assert_eq!(failing.failed_paths().count(), failing.paths.len());
        let text = format!("{}", failing);
        assert!(text.contains("0/1 paths converged"));
    }

    /// {x² − 2x + 1, y − x}: supports {(0,0),(1,0),(2,0)} × {(0,0),(1,0),
    /// (0,1)}, mixed volume 2, and the target has the **double root**
    /// (1, 1) — the smallest system whose paths end at a singular isolated
    /// root.
    fn double_root_system() -> MSystem<c64, 2, 2, 3> {
        let z = |v: f64| c64::new(v, 0.0);
        let f1 = MPoly::new(
            [z(1.0), z(-2.0), z(1.0)],
            [
                Monomial::new([0, 0]),
                Monomial::new([1, 0]),
                Monomial::new([2, 0]),
            ],
        );
        let f2 = MPoly::new(
            [z(0.0), z(-1.0), z(1.0)],
            [
                Monomial::new([0, 0]),
                Monomial::new([1, 0]),
                Monomial::new([0, 1]),
            ],
        );
        MSystem::new([f1, f2])
    }

    /// Endgame validation 1: the double root. Both paths must finish
    /// through the Cauchy endgame with winding 2 and land on (1, 1).
    ///
    /// The measured endpoint error is ~1e-15 (the Cauchy mean kills the
    /// √(1−t) Puiseux terms exactly, so the endpoint is as good as the
    /// circle samples) — dramatically better than the ~√ε ≈ 1e-8 a Newton
    /// polish can do at a double root; asserted at 1e-12 for margin.
    #[test]
    fn double_root_both_paths_end_singular_with_winding_2() {
        let system = double_root_system();
        let report = solve(&system, 1, &TrackOptions::default()).unwrap();
        assert_eq!(report.mixed_volume, 2);
        assert_eq!(report.paths.len(), 2);
        assert_eq!(report.converged_count(), 0);
        assert_eq!(report.singular_count(), 2);
        assert_eq!(report.failed_paths().count(), 0);

        let one = c64::new(1.0, 0.0);
        for p in &report.paths {
            assert_eq!(p.status, PathStatus::ConvergedSingular { winding: 2 });
            assert!(p.status.is_root());
            assert!(p.endgame_entered);
            assert_eq!(p.t_reached, 1.0);
            // Endpoint accuracy (measured ~9e-16; see the doc comment).
            let d = (p.point[0] - one)
                .magnitude()
                .max((p.point[1] - one).magnitude());
            assert!(d < 1e-12, "endpoint {:e} from (1,1)", d);
            // The residual gate the endgame applied: ‖H(ŷ,1)‖∞ ≤
            // √newton_tol·max(1,‖ŷ‖∞) = 1e-5·~1 (the actual residual is
            // ~1e-31: (x−1)² at x−1 ≈ 1e-15).
            assert!(residual_inf(&system, &p.point) < 1e-5);
        }
        // One distinct root of multiplicity 2 — both paths land on it.
        assert_eq!(report.distinct_solutions(1e-4).len(), 1);
        assert_eq!(report.solutions().count(), 2);
        assert_eq!(report.multiplicity_of(&[one, one], 1e-4), 2);
        assert_eq!(report.multiplicity_of(&[one + one, one], 1e-4), 0);

        // Display: the summary carries the singular tally and the per-path
        // lines the conv-singular tag.
        let text = format!("{}", report);
        assert!(text.contains("0/2 paths converged (+2 singular)"));
        assert!(text.contains("conv-singular"));
    }

    /// Endgame validation 2: the triple root {(x−1)³, y − 1} (mixed
    /// volume 3). All three paths form one winding-3 cycle; the measured
    /// endpoint error is ~2e-12 (cube-root conditioning is harsher than
    /// the double root's — the circle samples carry ~r^(1/3) structure —
    /// but still far inside the spec's 1e-4); asserted at 1e-9.
    #[test]
    fn triple_root_all_three_paths_wind_thrice() {
        let z = |v: f64| c64::new(v, 0.0);
        let f1 = MPoly::new(
            [z(-1.0), z(3.0), z(-3.0), z(1.0)],
            [
                Monomial::new([0, 0]),
                Monomial::new([1, 0]),
                Monomial::new([2, 0]),
                Monomial::new([3, 0]),
            ],
        );
        let f2 = MPoly::new(
            [z(-1.0), z(1.0), z(0.0), z(0.0)],
            [
                Monomial::new([0, 0]),
                Monomial::new([0, 1]),
                Monomial::new([0, 0]),
                Monomial::new([0, 0]),
            ],
        );
        let system = MSystem::new([f1, f2]);
        let report = solve(&system, 1, &TrackOptions::default()).unwrap();
        assert_eq!(report.mixed_volume, 3);
        assert_eq!(report.paths.len(), 3);
        assert_eq!(report.singular_count(), 3);
        assert_eq!(report.converged_count(), 0);

        let one = c64::new(1.0, 0.0);
        for p in &report.paths {
            assert_eq!(p.status, PathStatus::ConvergedSingular { winding: 3 });
            assert!(p.endgame_entered);
            let d = (p.point[0] - one)
                .magnitude()
                .max((p.point[1] - one).magnitude());
            assert!(d < 1e-9, "endpoint {:e} from (1,1)", d);
        }
        assert_eq!(report.distinct_solutions(1e-4).len(), 1);
        assert_eq!(report.multiplicity_of(&[one, one], 1e-4), 3);
    }

    /// Endgame negative control: {x² − 3x + 2, y − x} has the two *simple*
    /// roots (1,1) and (2,2) — same supports as the double-root system,
    /// but a regular target. Both paths must reach plain Converged without
    /// the endgame ever triggering (the endgame_entered flag is the
    /// proof).
    #[test]
    fn simple_roots_never_enter_the_endgame() {
        let z = |v: f64| c64::new(v, 0.0);
        let f1 = MPoly::new(
            [z(2.0), z(-3.0), z(1.0)],
            [
                Monomial::new([0, 0]),
                Monomial::new([1, 0]),
                Monomial::new([2, 0]),
            ],
        );
        let f2 = MPoly::new(
            [z(0.0), z(-1.0), z(1.0)],
            [
                Monomial::new([0, 0]),
                Monomial::new([1, 0]),
                Monomial::new([0, 1]),
            ],
        );
        let system = MSystem::new([f1, f2]);
        let report = solve(&system, 1, &TrackOptions::default()).unwrap();
        assert_eq!(report.mixed_volume, 2);
        assert_eq!(report.converged_count(), 2);
        assert_eq!(report.singular_count(), 0);
        for p in &report.paths {
            assert_eq!(p.status, PathStatus::Converged);
            assert!(!p.endgame_entered, "endgame triggered on a simple root");
        }
        for x in [1.0, 2.0] {
            let pt = [c64::new(x, 0.0), c64::new(x, 0.0)];
            assert_eq!(report.multiplicity_of(&pt, 1e-6), 1);
        }
        assert_eq!(report.distinct_solutions(1e-6).len(), 2);
    }

    /// Endgame validation 3: the double root over `Complex<f32>`, with
    /// every tolerance loosened to single precision. The f64 trigger
    /// constants starve here: f32's cancellation zone around the double
    /// root is `~√ε_f32 ≈ 4e-4` wide, so the tracker cruises to `t = 1`
    /// with healthy-looking `dt` and pivots around `1e-4` — the pivot
    /// threshold must sit above that scale and the dt-collapse condition
    /// must be disarmed for the terminal-accept trigger to see it.
    /// Winding detection must still work; the measured endpoint error is
    /// ~1e-7 (vs the ~2e-4 plain Newton left), asserted at 1e-3.
    #[test]
    fn double_root_in_f32() {
        use crate::complex::c32;
        let z = |v: f32| c32::new(v, 0.0);
        let f1 = MPoly::new(
            [z(1.0), z(-2.0), z(1.0)],
            [
                Monomial::new([0, 0]),
                Monomial::new([1, 0]),
                Monomial::new([2, 0]),
            ],
        );
        let f2 = MPoly::new(
            [z(0.0), z(-1.0), z(1.0)],
            [
                Monomial::new([0, 0]),
                Monomial::new([1, 0]),
                Monomial::new([0, 1]),
            ],
        );
        let system = MSystem::new([f1, f2]);
        let options = TrackOptions::<f32> {
            newton_tol: 1e-5,
            dt_min: 1e-6,
            endgame_radius: 1e-2,
            endgame_closure_tol: 1e-3,
            endgame_pivot_threshold: 1e-2,
            endgame_dt_threshold: 1.0, // pivot collapse alone decides
            ..TrackOptions::default()
        };
        let report = solve(&system, 1, &options).unwrap();
        assert_eq!(report.mixed_volume, 2);
        assert_eq!(report.singular_count(), 2);
        let one = c32::new(1.0, 0.0);
        for p in &report.paths {
            assert_eq!(p.status, PathStatus::ConvergedSingular { winding: 2 });
            let d = (p.point[0] - one)
                .magnitude()
                .max((p.point[1] - one).magnitude());
            assert!(d < 1e-3, "f32 endpoint {:e} from (1,1)", d);
        }
        assert_eq!(report.multiplicity_of(&[one, one], 1e-2), 2);
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
