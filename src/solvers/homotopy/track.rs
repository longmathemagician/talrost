//! Per-cell homotopies and the predictor–corrector path tracker — the
//! **online** phase.
//!
//! For a target system `F_i(x) = Σ_{a ∈ A_i} c_{i,a}·x^a`, a lifting `ω`,
//! and a fine mixed cell with normal `α` and edge `(a_i, b_i)` per equation,
//! the Huber–Sturmfels cell homotopy is
//!
//! ```text
//! H_i(y, t) = Σ_a c_{i,a} · t^{e_{i,a}} · y^a,        t ∈ [0, 1],
//! e_{i,a}   = ⟨a, α⟩ + ω_i(a) − m_i,                  m_i = ⟨a_i, α⟩ + ω_i(a_i).
//! ```
//!
//! `m_i` is the shared level of the two edge points, so `e` is **exactly
//! zero** for both edge terms and strictly positive (generically) for every
//! other term. Consequences:
//!
//! - at `t = 0` only the edge terms survive: `H(·, 0)` **is** the binomial
//!   start system that [`super::start::start_solutions`] solves;
//! - at `t = 1` every `t^e` is one: `H(·, 1)` **is** `F`. No coordinate
//!   transform is needed at either endpoint — the tracker follows `y`
//!   directly and its `t = 1` point is a root of the target.
//!
//! On top of this, [`CellHomotopy`] rotates every non-edge term by the
//! endpoint-preserving phase `exp(iγe(1−t))` — the **γ-twist**, the gamma
//! trick that keeps real-coefficient targets (whose plain coefficient paths
//! `c·t^e` never leave the real slice) from folding on the discriminant
//! mid-path; see [`CellHomotopy`] for the geometry.
//!
//! [`track_path`] continues one start root from `t = 0` to `t = 1` with a
//! Runge–Kutta predictor ([`Predictor`]: Euler, midpoint RK2, or classical
//! RK4, each stage a tangent solve `J·ẏ = −∂H/∂t`) and a Newton corrector
//! at fixed `t`, under corrector-informed step control ([`TrackOptions`]).
//!
//! # The `t = 0` tangent singularity (and the corrector-only first step)
//!
//! Coefficient-wise, `d/dt [c·t^e] = c·e·t^{e−1}` — constant terms
//! (`e = 0`) differentiate to zero, but fractional levels `0 < e < 1` make
//! `t^{e−1}` blow up as `t → 0⁺`, so the tangent must **never** be
//! evaluated at `t = 0` exactly. The tracker therefore takes its first step
//! without a predictor: it steps `t` from `0` to `dt` and corrects with
//! Newton at fixed `t = dt`, starting from the exact start root. This
//! "corrector-only first step" is the standard resolution and sidesteps the
//! singular derivative entirely; every later step has `t > 0` and predicts
//! normally.
//!
//! Everything here is **allocation-free by construction** — fixed-size
//! arrays and `Copy` types only, no `Vec` — even though the module currently
//! ships inside the `std`-gated [`super`] tree: a future `no_std` exposure
//! of the online half (offline data baked at build time) only needs the
//! gate moved, not the code changed.

use crate::complex::Complex;
use crate::matrix::Matrix;
use crate::mvpoly::{MSystem, Monomial};
use crate::real::Real;
use crate::scalar::Scalar;
use crate::vector::Vector;

use super::cells::MixedCell;
use super::support::{Lifting, Support};

/// Exact conversion of a small exponent to `F` (the same contract as the
/// offline enumeration: sparse exponents never approach 2²⁴).
fn exp_to_f<F: Real>(e: i32) -> F {
    let mag = F::from_u32(e.unsigned_abs());
    if e < 0 {
        -mag
    } else {
        mag
    }
}

/// `10^k` in `F`, for the decimal defaults of [`TrackOptions`].
fn tenpow<F: Real>(k: i32) -> F {
    F::from_u32(10).powi(k)
}

/// The ∞-norm of a complex point: `max_i |y_i|`.
fn inf_norm<F: Real, const NV: usize>(y: &[Complex<F>; NV]) -> F {
    let mut m = F::ZERO;
    for yi in y {
        m = m.max(yi.magnitude());
    }
    m
}

/// `y + k·s` componentwise — the predictor's stage-advance primitive.
fn add_scaled<F: Real, const NV: usize>(
    y: &[Complex<F>; NV],
    k: &[Complex<F>; NV],
    s: F,
) -> [Complex<F>; NV] {
    let mut out = *y;
    for (o, ki) in out.iter_mut().zip(k.iter()) {
        *o += *ki * s;
    }
    out
}

/// The path tangent `ẏ` at `(y, t)`: solves the Davidenko system
/// `J(y, t)·ẏ = −H_t(y, t)` with a fresh Jacobian and LU factorization
/// (`work` is the reusable coefficient buffer). `None` when the Jacobian is
/// singular to working precision. Must not be called at `t = 0` (see
/// [`CellHomotopy::dt`]).
fn tangent<F: Real, const NV: usize, const MAXT: usize>(
    homotopy: &CellHomotopy<F, NV, MAXT>,
    work: &mut MSystem<Complex<F>, NV, NV, MAXT>,
    y: &[Complex<F>; NV],
    t: F,
) -> Option<[Complex<F>; NV]> {
    homotopy.write_system_at(t, work);
    let (_, j) = work.eval_jacobian(y);
    let lu = j.lu()?;
    let ht = homotopy.dt(y, t);
    Some(lu.solve(&Vector::new(ht.map(|h| -h))).b)
}

/// One cell's homotopy `H_i(y, t) = Σ_a c_{i,a}·t^{e_{i,a}}·φ_{i,a}(t)·y^a`,
/// with the target coefficients `c` and shifted levels `e` precomputed from
/// `(system, lifting, cell)` — see the [module docs](self) for the math —
/// and `φ` the γ-twist phase described below.
///
/// The system is square (`NEQ == NV`): a mixed cell supplies exactly one
/// edge per variable and the tracker's Newton corrector needs a square
/// Jacobian for its LU solve, so a separate `NEQ` parameter would only
/// admit unsatisfiable instantiations.
///
/// # The γ-twist (discriminant avoidance for non-generic targets)
///
/// The plain Huber–Sturmfels homotopy multiplies each coefficient by the
/// **real positive** factor `t^e`, so a target with real coefficients stays
/// a real system for every `t` and every lifting. The discriminant has real
/// codimension **one** inside that real slice, so a real one-parameter
/// family generically crosses it: two conjugate roots collide at a fold and
/// the Jacobian turns singular *mid-path* — exactly what happens on
/// cyclic-3, whose six paths all stall at one interior `t` in conjugate
/// pairs. Smoothness of the paths on `t ∈ (0, 1)` is only guaranteed for
/// *generic complex* coefficients; symmetric real benchmarks are as
/// non-generic as it gets.
///
/// The classical cure is the **gamma trick** (Sommese–Wampler), applied
/// here at the homotopy level: every non-edge term is additionally rotated
/// by the phase
///
/// ```text
/// φ(t) = exp(i·γ·e·(1 − t)),
/// ```
///
/// which is exactly `1` at `t = 1` and irrelevant at `t = 0` (the term
/// vanishes there), so **both endpoints are untouched** — `H(·, 0)` is
/// still the binomial start and `H(·, 1)` still the target, bit-for-bit.
/// Mid-path, the coefficients leave the real slice: the family now moves
/// through complex coefficient space, where the discriminant has real
/// codimension two, and misses it for all but a measure-zero set of `γ`.
/// This is a reparametrization of the *coefficient arc*, not of the path
/// set alone — the tracked paths are different curves with the same
/// endpoints. `γ = 0` recovers the untwisted homotopy
/// ([`CellHomotopy::with_gamma`]).
///
/// All per-step methods are allocation-free; the write-into-buffer variant
/// [`CellHomotopy::write_system_at`] lets a tracker reuse one [`MSystem`]
/// workspace across steps instead of copying support layouts each time.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct CellHomotopy<F: Real, const NV: usize, const MAXT: usize> {
    /// The target system (`t = 1` coefficients plus the support layout every
    /// buffer written by this homotopy shares).
    target: MSystem<Complex<F>, NV, NV, MAXT>,
    /// `levels[i][k]` = the shifted level `e` of term `k` of equation `i`:
    /// exactly zero for the cell's edge terms (and for padding), strictly
    /// positive otherwise.
    levels: [[F; MAXT]; NV],
    /// The γ-twist angle (radians); `0` disables the twist.
    gamma: F,
}

impl<F: Real, const NV: usize, const MAXT: usize> CellHomotopy<F, NV, MAXT> {
    /// Precomputes the homotopy of `cell` for `system`.
    ///
    /// `supports`/`liftings` must be the pair `cell` was enumerated from
    /// (with `supports == Support::from_msystem(system)`), since lift values
    /// are looked up by support-point order. The edge terms' levels are set
    /// to exactly zero (they are level by construction; recomputing them
    /// would leave float dust), and every other term's level is measured
    /// from the smaller of the two edge levels, so it stays strictly
    /// positive.
    ///
    /// # Level normalization (a reparametrization, not a new homotopy)
    ///
    /// Raw lifted levels can be tiny — lift values live in `[0, 1)`, so a
    /// margin like `e = 0.07` is routine — and then `t^e` stays near 1 for
    /// every reachable `t` (`0.01^0.07 ≈ 0.72`): the homotopy would only
    /// approach its binomial start below `t ≈ 1e-14`, unreachable stiffness.
    /// Since a tracker only cares about the *path set*, `new` substitutes
    /// `t ↦ t^(1/e_min)` — i.e. divides **all** levels by the smallest
    /// positive one, the same global factor for every equation — which
    /// traverses exactly the same paths at a different speed and leaves both
    /// endpoints untouched. After normalization the smallest positive level
    /// is exactly 1, so the start perturbation is `O(t)`.
    ///
    /// # Panics
    ///
    /// Panics if some `liftings[i]` does not have exactly one value per
    /// point of `supports[i]`, or if a nonzero-coefficient term of `system`
    /// is missing from its support (the supports belong to a different
    /// system).
    pub fn new(
        system: &MSystem<Complex<F>, NV, NV, MAXT>,
        supports: &[Support<NV>; NV],
        liftings: &[Lifting<F>; NV],
        cell: &MixedCell<F, NV>,
    ) -> Self {
        // ln 2 — an arbitrary fixed angle with no resonance with the
        // roots-of-unity structure of binomial starts (see the type-level
        // γ-twist docs; any "random enough" angle works, and a fixed one
        // keeps same-seed solves bit-for-bit reproducible).
        Self::with_gamma(system, supports, liftings, cell, F::from_u32(2).ln())
    }

    /// [`CellHomotopy::new`] with an explicit γ-twist angle (radians);
    /// `gamma = 0` gives the untwisted textbook homotopy `c·t^e·y^a` —
    /// adequate for generic complex coefficients, fold-prone for real ones
    /// (see the type-level docs).
    pub fn with_gamma(
        system: &MSystem<Complex<F>, NV, NV, MAXT>,
        supports: &[Support<NV>; NV],
        liftings: &[Lifting<F>; NV],
        cell: &MixedCell<F, NV>,
        gamma: F,
    ) -> Self {
        let mut levels = [[F::ZERO; MAXT]; NV];
        for ((((lrow, poly), sup), lift), &(a, b)) in levels
            .iter_mut()
            .zip(system.polys.iter())
            .zip(supports.iter())
            .zip(liftings.iter())
            .zip(cell.edges.iter())
        {
            assert!(
                sup.len() == lift.len(),
                "CellHomotopy: lifting has {} values for {} support points",
                lift.len(),
                sup.len()
            );
            // The lifted level ⟨m, α⟩ + ω(m) of a support point.
            let level_of = |m: &Monomial<NV>| -> F {
                let j = sup
                    .points()
                    .iter()
                    .position(|p| p == m)
                    .expect("CellHomotopy: system term missing from its support");
                let mut acc = lift.values()[j];
                for (&e, &al) in m.exps.iter().zip(cell.normal.iter()) {
                    acc += exp_to_f::<F>(e) * al;
                }
                acc
            };
            let edge_level = level_of(&a).min(level_of(&b));
            for ((lv, coeff), mono) in lrow
                .iter_mut()
                .zip(poly.coeffs.iter())
                .zip(poly.support.iter())
            {
                *lv = if coeff.norm_sqr() == F::ZERO || *mono == a || *mono == b {
                    // Padding never contributes (and its exponents are
                    // garbage — don't look them up); edge terms are level
                    // by construction.
                    F::ZERO
                } else {
                    let e = level_of(mono) - edge_level;
                    debug_assert!(
                        e > F::ZERO,
                        "CellHomotopy: non-edge term at or below the edge level \
                         (cell/lifting mismatch?)"
                    );
                    e
                };
            }
        }

        // Normalize levels by the smallest positive one (see the doc
        // comment above): min positive level becomes exactly 1. A system
        // whose every term is an edge term (pure binomial target) has no
        // positive level and needs no normalization.
        let mut e_min = F::INFINITY;
        for lrow in levels.iter() {
            for &e in lrow.iter() {
                if e > F::ZERO && e < e_min {
                    e_min = e;
                }
            }
        }
        if e_min < F::INFINITY {
            for lrow in levels.iter_mut() {
                for e in lrow.iter_mut() {
                    *e /= e_min;
                }
            }
        }

        Self {
            target: *system,
            levels,
            gamma,
        }
    }

    /// The target system `F = H(·, 1)` — the exact `t = 1` coefficients,
    /// for the tracker's terminal Newton polish.
    pub fn target(&self) -> &MSystem<Complex<F>, NV, NV, MAXT> {
        &self.target
    }

    /// The γ-twist angle this homotopy was built with (radians; see the
    /// type-level docs). [`CellHomotopy::new`] uses `ln 2`.
    pub fn gamma(&self) -> F {
        self.gamma
    }

    /// Writes the coefficients of `H(·, t)` — `c·t^e·exp(iγe(1−t))` per
    /// term (the γ-twist phase, see the type-level docs) — into `sys`,
    /// leaving its support untouched. `sys` must share the target's support
    /// layout (any system produced by [`CellHomotopy::system_at`] does);
    /// this is the buffer-reuse fast path for trackers.
    ///
    /// Endpoints are exact: `e = 0` terms keep the coefficient `c` verbatim
    /// at every `t` (no `powf` roundoff), and at `t = 1` both `t^e` and the
    /// twist phase are exactly one, so `t = 0` yields the binomial start
    /// system and `t = 1` yields `F` bit-for-bit.
    pub fn write_system_at(&self, t: F, sys: &mut MSystem<Complex<F>, NV, NV, MAXT>) {
        let one_minus_t = F::ONE - t;
        for ((poly, tpoly), lrow) in sys
            .polys
            .iter_mut()
            .zip(self.target.polys.iter())
            .zip(self.levels.iter())
        {
            debug_assert!(
                poly.support == tpoly.support,
                "write_system_at: buffer support layout differs from the target's"
            );
            for ((c, &tc), &e) in poly
                .coeffs
                .iter_mut()
                .zip(tpoly.coeffs.iter())
                .zip(lrow.iter())
            {
                *c = if e == F::ZERO {
                    tc
                } else {
                    tc * t.powf(e) * Complex::from_polar(F::ONE, self.gamma * e * one_minus_t)
                };
            }
        }
    }

    /// The system `H(·, t)` as a fresh [`MSystem`] (stack copy of the
    /// target's support plus [`CellHomotopy::write_system_at`]
    /// coefficients).
    pub fn system_at(&self, t: F) -> MSystem<Complex<F>, NV, NV, MAXT> {
        let mut sys = self.target;
        self.write_system_at(t, &mut sys);
        sys
    }

    /// `H(y, t)`.
    pub fn eval(&self, y: &[Complex<F>; NV], t: F) -> [Complex<F>; NV] {
        self.system_at(t).eval(y)
    }

    /// `H(y, t)` and the Jacobian `∂H/∂y` in one forward-AD sweep
    /// (delegates to [`MSystem::eval_jacobian`] with the `c·t^e`
    /// coefficients).
    #[allow(clippy::type_complexity)]
    pub fn eval_jacobian(
        &self,
        y: &[Complex<F>; NV],
        t: F,
    ) -> ([Complex<F>; NV], Matrix<Complex<F>, NV, NV>) {
        self.system_at(t).eval_jacobian(y)
    }

    /// The t-derivative `∂H/∂t (y, t)`: evaluation with the coefficient set
    /// `c·e·t^{e−1}·exp(iγe(1−t))·(1 − iγt)` — the product rule over
    /// `t^e·φ(t)` with the γ-twist phase `φ` (constant `e = 0` terms
    /// differentiate to zero, and `γ = 0` collapses to the textbook
    /// `c·e·t^{e−1}`).
    ///
    /// Must not be called at `t = 0`: fractional levels `0 < e < 1` make
    /// `t^{e−1}` singular there (`debug_assert`ed) — the reason the
    /// tracker's first step is corrector-only (see the [module
    /// docs](self)).
    pub fn dt(&self, y: &[Complex<F>; NV], t: F) -> [Complex<F>; NV] {
        debug_assert!(
            t > F::ZERO,
            "CellHomotopy::dt: t^(e-1) is singular at t = 0 — take a corrector-only first step"
        );
        let one_minus_t = F::ONE - t;
        // d/dt [t^e·e^{iγe(1−t)}] = e·t^{e−1}·e^{iγe(1−t)}·(1 − iγt).
        let chain = Complex::new(F::ONE, -(self.gamma * t));
        let mut sys = self.target;
        for (poly, lrow) in sys.polys.iter_mut().zip(self.levels.iter()) {
            for (c, &e) in poly.coeffs.iter_mut().zip(lrow.iter()) {
                *c = if e == F::ZERO {
                    Complex::new(F::ZERO, F::ZERO)
                } else {
                    *c * (e * t.powf(e - F::ONE))
                        * Complex::from_polar(F::ONE, self.gamma * e * one_minus_t)
                        * chain
                };
            }
        }
        sys.eval(y)
    }
}

/// The predictor scheme of [`track_path`]: how the trial point for the next
/// `t` is extrapolated before Newton correction. Every scheme integrates the
/// Davidenko ODE `ẏ = −J(y, t)⁻¹·H_t(y, t)` across one step; each tangent
/// stage costs one Jacobian build plus a **fresh LU factorization** (the
/// Jacobian changes with the stage point), so the per-step cost is 1/2/4
/// tangent solves for Euler/RK2/RK4 while the local truncation error drops
/// as `O(h²)`/`O(h³)`/`O(h⁵)` — higher orders hand the corrector a much
/// better trial point and let the step control take far fewer, larger steps.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Predictor {
    /// First-order Euler: one tangent at `(y, t)`.
    Euler,
    /// Second-order midpoint rule (RK2): a half-step Euler stage, then the
    /// midpoint tangent carries the whole step.
    Rk2,
    /// Classical fourth-order Runge–Kutta: four tangent stages, combined
    /// with the 1/6–2/6–2/6–1/6 weights.
    Rk4,
}

impl Default for Predictor {
    /// [`Predictor::Rk4`] — the measured best, by a wide margin. Full
    /// `solve()` release measurements under the corrector-informed step
    /// control (total steps summed over all paths, deterministic per seed;
    /// wall times from the Phase 9 dev machine — run
    /// `cargo run --release --example bench` to reproduce):
    ///
    /// | system (seed)          | Euler            | RK2              | RK4             |
    /// |------------------------|------------------|------------------|-----------------|
    /// | dense conic, MV 4 (4)  | 3654 st / 8.6 ms | 1462 st / 5.0 ms | 186 st / 1.0 ms |
    /// | cyclic-3, MV 6 (1)     | 3891 st / 8.6 ms | 1152 st / 3.5 ms | 210 st / 1.0 ms |
    ///
    /// The step control only grows `dt` when the corrector converges in a
    /// single Newton iteration, which at the default `newton_tol = 1e-10`
    /// demands a trial point already accurate to ~1e-10: RK4's `O(dt⁵)`
    /// local error meets that at usable step sizes, while the low-order
    /// predictors almost never do — their `dt` can only drift down, hence
    /// the ~20× step gap. RK4's four-fold tangent cost per step is repaid
    /// roughly eight-fold in wall time.
    fn default() -> Self {
        Predictor::Rk4
    }
}

/// Step-control and convergence knobs for [`track_path`]. `Copy`, with
/// [`Default`] values tuned for `f64` (an `f32` instantiation would need a
/// much looser `newton_tol` than its default `1e-10`).
///
/// # Step control (corrector-informed)
///
/// After an **accepted** step, `dt` adapts to the Newton effort the
/// corrector actually spent: converged in 1 iteration → `dt *= grow`
/// (capped at `dt_max`); 2 iterations → `dt` unchanged; converged only on
/// the `max_newton`-th iteration → `dt *= 0.8` — the corrector is straining,
/// so back off *before* it starts failing (iteration counts strictly between
/// 2 and `max_newton` also leave `dt` unchanged). A **rejected** step
/// (corrector failure or singular Jacobian) multiplies `dt` by `shrink` and
/// retries from the saved point; `dt < dt_min` aborts the path. The last
/// step is always clamped to land exactly on `t = 1`.
///
/// The growth condition couples to the predictor order: converging in one
/// iteration needs a trial point already within `newton_tol`, which at the
/// tight default tolerance only [`Predictor::Rk4`] delivers at useful step
/// sizes — under Euler or RK2 the step size mostly drifts down and tracking
/// takes an order of magnitude more steps (measured on the [`Predictor`]
/// benchmark systems). Pair this rule with RK4, or loosen `newton_tol`.
///
/// # Stability
///
/// This struct is deliberately **not** `#[non_exhaustive]`: the crate is
/// pre-1.0 and new knobs are expected, and keeping plain struct syntax lets
/// callers write `TrackOptions { max_newton: 5, ..Default::default() }` —
/// which `#[non_exhaustive]` would forbid outside this crate. New fields are
/// an accepted breaking change until 1.0.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct TrackOptions<F> {
    /// Predictor scheme (default [`Predictor::Rk4`]; see [`Predictor`] for
    /// the cost/accuracy trade and the benchmark that chose the default).
    pub predictor: Predictor,
    /// Initial step size in `t` (default `1e-2`).
    pub dt_init: F,
    /// Smallest allowed step: shrinking below this aborts the path with
    /// [`PathStatus::MinStepReached`] (default `1e-14`).
    pub dt_min: F,
    /// Largest allowed step (default `1e-1`).
    pub dt_max: F,
    /// Newton convergence tolerance, applied to the ∞-norm of the Newton
    /// update relative to `max(1, ‖y‖∞)` (default `1e-10`).
    pub newton_tol: F,
    /// Newton iterations per corrector attempt (default 3).
    pub max_newton: u32,
    /// Attempted steps (accepted + rejected) before giving up with
    /// [`PathStatus::MaxStepsReached`] (default 10 000).
    pub max_steps: u32,
    /// The path is declared [`PathStatus::Diverged`] when `‖y‖∞` exceeds
    /// this bound (default `1e8`).
    pub divergence_bound: F,
    /// Step growth factor after a step whose corrector converged in a
    /// single Newton iteration (default 2.0) — see the step-control notes
    /// on [`TrackOptions`].
    pub grow: F,
    /// Step shrink factor after a rejected step (default 0.5).
    pub shrink: F,
}

impl<F: Real> Default for TrackOptions<F> {
    fn default() -> Self {
        Self {
            predictor: Predictor::default(),
            dt_init: tenpow(-2),
            dt_min: tenpow(-14),
            dt_max: tenpow(-1),
            newton_tol: tenpow(-10),
            max_newton: 3,
            max_steps: 10_000,
            divergence_bound: tenpow(8),
            grow: F::from_u32(2),
            shrink: F::ONE / F::from_u32(2),
        }
    }
}

/// How a tracked path ended.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PathStatus {
    /// Reached `t = 1` with a converged corrector and a final Newton polish
    /// against the target system: the endpoint approximates a root of `F`.
    Converged,
    /// Step halving pushed `dt` below [`TrackOptions::dt_min`] with the
    /// corrector still failing its update-norm test.
    MinStepReached,
    /// [`TrackOptions::max_steps`] attempted steps (accepted + rejected)
    /// without reaching `t = 1`.
    MaxStepsReached,
    /// The LU factorization of `∂H/∂y` failed during prediction or
    /// correction, and the dt-halving retry loop also ran out of step: the
    /// Jacobian is singular to working precision along the path.
    SingularJacobian,
    /// `‖y‖∞` exceeded [`TrackOptions::divergence_bound`] — the path is
    /// heading to infinity (a root of the target at infinity, or outside
    /// the torus).
    Diverged,
}

impl core::fmt::Display for PathStatus {
    /// Short lowercase tags — `converged`, `min-step`, `max-steps`,
    /// `singular`, `diverged` — honoring width/alignment flags (via
    /// [`core::fmt::Formatter::pad`]) so reports can column-align them.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.pad(match self {
            PathStatus::Converged => "converged",
            PathStatus::MinStepReached => "min-step",
            PathStatus::MaxStepsReached => "max-steps",
            PathStatus::SingularJacobian => "singular",
            PathStatus::Diverged => "diverged",
        })
    }
}

/// The outcome of tracking one path: where it got to and how.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct PathResult<F: Real, const NV: usize> {
    /// The final point: a polished root of the target when `status` is
    /// [`PathStatus::Converged`], otherwise the last accepted point.
    pub point: [Complex<F>; NV],
    /// The last accepted `t` (`1` exactly on convergence).
    pub t_reached: F,
    /// Why tracking stopped.
    pub status: PathStatus,
    /// Attempted predictor–corrector steps, accepted and rejected.
    pub steps: u32,
    /// Total Newton iterations across all correctors and the final polish.
    pub newton_iters: u32,
    /// The pivot-norm ratio `min|U_ii| / max|U_ii|` of the most recent
    /// successful Newton factorization ([`crate::matrix::Lu::pivot_ratio`]):
    /// for a [`PathStatus::Converged`] path this comes from the **final
    /// polished Newton solve against the target system**, for a failed path
    /// from its last accepted corrector, and it is `0` when no Newton solve
    /// ever succeeded. A cheap singularity-proximity signal — `0` means
    /// singular, values near `1` mean well-balanced pivots — **not** a
    /// condition number: see [`crate::matrix::Lu::pivot_ratio`] for the
    /// caveats.
    pub pivot_ratio: F,
}

/// Tracks one path of `homotopy` from the start root `start` (a solution of
/// `H(·, 0)`, the cell's binomial system) to `t = 1`.
///
/// Each step attempts: **predict** — integrate the Davidenko ODE
/// `J(y, t)·ẏ = −∂H/∂t` from `t` to `t + dt` with the configured
/// [`Predictor`] (one to four tangent solves; skipped on the very first
/// step, which is corrector-only — see the [module docs](self) for why
/// `t = 0` forbids the tangent); **correct** — up to
/// [`TrackOptions::max_newton`] Newton iterations at fixed `t + dt`,
/// accepting when the update ∞-norm falls below
/// `newton_tol · max(1, ‖y‖∞)`. Accepted steps adapt `dt` to the observed
/// Newton effort and rejected steps shrink it and retry from the saved
/// point (see the step-control notes on [`TrackOptions`]); the last step is
/// clamped to land exactly on `t = 1`. There the endpoint gets a final
/// Newton polish against the target system itself (best-effort: a singular
/// Jacobian there keeps the corrector-converged point) and the path reports
/// [`PathStatus::Converged`] together with the polish's
/// [`PathResult::pivot_ratio`] conditioning hint.
///
/// Allocation-free: the only working state is a handful of stack arrays and
/// one [`MSystem`] coefficient buffer.
pub fn track_path<F: Real, const NV: usize, const MAXT: usize>(
    homotopy: &CellHomotopy<F, NV, MAXT>,
    start: [Complex<F>; NV],
    options: &TrackOptions<F>,
) -> PathResult<F, NV> {
    let two = F::from_u32(2);
    let six = F::from_u32(6);
    // The "accepted, but only just" backoff factor: 0.8 (see TrackOptions).
    let soft_shrink = F::from_u32(4) / F::from_u32(5);

    let mut y = start;
    let mut t = F::ZERO;
    let mut dt = options.dt_init.min(options.dt_max);
    let mut steps = 0u32;
    let mut newton_iters = 0u32;
    // Conditioning hint of the last successful Newton factorization; stays
    // 0 until a corrector accepts (see PathResult::pivot_ratio).
    let mut pivot_ratio = F::ZERO;
    // No tangent exists at t = 0 (see module docs): the first accepted step
    // is corrector-only, and every retry of it stays corrector-only.
    let mut tangent_ready = false;
    // The reusable coefficient buffer (support layout fixed by the target).
    let mut work = homotopy.system_at(F::ZERO);

    loop {
        if inf_norm(&y) > options.divergence_bound {
            return PathResult {
                point: y,
                t_reached: t,
                status: PathStatus::Diverged,
                steps,
                newton_iters,
                pivot_ratio,
            };
        }
        if t >= F::ONE {
            break; // reached the end; polish below
        }
        if steps >= options.max_steps {
            return PathResult {
                point: y,
                t_reached: t,
                status: PathStatus::MaxStepsReached,
                steps,
                newton_iters,
                pivot_ratio,
            };
        }
        steps += 1;

        // Clamp the target time so the last step lands exactly on t = 1.
        let t_next = if t + dt >= F::ONE { F::ONE } else { t + dt };
        let mut y_trial = y;
        let mut singular = false;

        // Predict: integrate ẏ = −J(y, t)⁻¹·H_t(y, t) from t to t_next.
        // Every stage time is ≥ t > 0 (the first step is corrector-only),
        // so the t = 0 tangent singularity is never touched.
        if tangent_ready {
            let h = t_next - t;
            let half = h / two;
            let predicted = match options.predictor {
                Predictor::Euler => {
                    tangent(homotopy, &mut work, &y, t).map(|k1| add_scaled(&y, &k1, h))
                }
                Predictor::Rk2 => tangent(homotopy, &mut work, &y, t).and_then(|k1| {
                    let y2 = add_scaled(&y, &k1, half);
                    tangent(homotopy, &mut work, &y2, t + half).map(|k2| add_scaled(&y, &k2, h))
                }),
                Predictor::Rk4 => (|| {
                    let k1 = tangent(homotopy, &mut work, &y, t)?;
                    let k2 = tangent(homotopy, &mut work, &add_scaled(&y, &k1, half), t + half)?;
                    let k3 = tangent(homotopy, &mut work, &add_scaled(&y, &k2, half), t + half)?;
                    let k4 = tangent(homotopy, &mut work, &add_scaled(&y, &k3, h), t_next)?;
                    // y + (k1 + 2·k2 + 2·k3 + k4)·h/6.
                    let mut acc = k1;
                    for (((a, &b), &c), &d) in
                        acc.iter_mut().zip(k2.iter()).zip(k3.iter()).zip(k4.iter())
                    {
                        *a += (b + c) * two + d;
                    }
                    Some(add_scaled(&y, &acc, h / six))
                })(),
            };
            match predicted {
                Some(p) => y_trial = p,
                None => singular = true,
            }
        }

        // Correct: Newton at fixed t_next, counting the iterations this
        // attempt actually needed (the step-control signal).
        let mut converged = false;
        let mut used = 0u32;
        let mut ratio = F::ZERO;
        if !singular {
            homotopy.write_system_at(t_next, &mut work);
            for _ in 0..options.max_newton {
                let (h, j) = work.eval_jacobian(&y_trial);
                let Some(lu) = j.lu() else {
                    singular = true;
                    break;
                };
                newton_iters += 1;
                used += 1;
                ratio = lu.pivot_ratio();
                let delta = lu.solve(&Vector::new(h.map(|hi| -hi)));
                for (yt, d) in y_trial.iter_mut().zip(delta.b.iter()) {
                    *yt += *d;
                }
                if inf_norm(&delta.b) <= options.newton_tol * F::ONE.max(inf_norm(&y_trial)) {
                    converged = true;
                    break;
                }
            }
        }

        if converged {
            y = y_trial;
            t = t_next;
            tangent_ready = true;
            pivot_ratio = ratio;
            // Corrector-informed step control (see TrackOptions): grow on
            // a 1-iteration accept, back off softly when the corrector only
            // just made it, hold otherwise.
            if used <= 1 {
                dt = (dt * options.grow).min(options.dt_max);
            } else if used >= options.max_newton {
                dt *= soft_shrink;
            }
        } else {
            // Reject: restore is implicit (y was never overwritten), halve.
            dt *= options.shrink;
            if dt < options.dt_min {
                return PathResult {
                    point: y,
                    t_reached: t,
                    status: if singular {
                        PathStatus::SingularJacobian
                    } else {
                        PathStatus::MinStepReached
                    },
                    steps,
                    newton_iters,
                    pivot_ratio,
                };
            }
        }
    }

    // Terminal polish against the target coefficients themselves. H(·, 1)
    // is F exactly (write_system_at keeps e = 0 terms verbatim and
    // 1^e = 1), so this only squeezes the last corrector's roundoff out;
    // best-effort by design — a singular Jacobian at the root (e.g. a
    // multiple root) keeps the corrector-converged point. The last
    // factorization here is what PathResult::pivot_ratio reports.
    for _ in 0..options.max_newton {
        let (h, j) = homotopy.target().eval_jacobian(&y);
        let Some(lu) = j.lu() else { break };
        newton_iters += 1;
        pivot_ratio = lu.pivot_ratio();
        let delta = lu.solve(&Vector::new(h.map(|hi| -hi)));
        for (yi, d) in y.iter_mut().zip(delta.b.iter()) {
            *yi += *d;
        }
        if inf_norm(&delta.b) <= options.newton_tol * F::ONE.max(inf_norm(&y)) {
            break;
        }
    }

    PathResult {
        point: y,
        t_reached: F::ONE,
        status: PathStatus::Converged,
        steps,
        newton_iters,
        pivot_ratio,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::complex::c64;
    use crate::mvpoly::MPoly;
    use crate::solvers::homotopy::start::start_solutions;
    use crate::solvers::homotopy::support::random_liftings;
    use crate::solvers::homotopy::{mixed_cells, Support};

    /// The mixed-volume-2 trinomial pair from the offline tests, with fixed
    /// complex coefficients.
    fn trinomial() -> MSystem<c64, 2, 2, 3> {
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
        MSystem::new([f1, f2])
    }

    fn residual_inf(sys: &MSystem<c64, 2, 2, 3>, x: &[c64; 2]) -> f64 {
        let h = sys.eval(x);
        h[0].magnitude().max(h[1].magnitude())
    }

    #[test]
    fn homotopy_endpoints_are_start_and_target() {
        let system = trinomial();
        let supports = Support::from_msystem(&system);
        let liftings = random_liftings::<f64, 2>(&supports, 2026);
        let cells = mixed_cells(&supports, &liftings).unwrap();

        for cell in &cells {
            let hom = CellHomotopy::new(&system, &supports, &liftings, cell);

            // t = 1: exactly the target, bit-for-bit.
            assert_eq!(hom.system_at(1.0), system);
            assert_eq!(hom.target(), &system);

            // t = 0: exactly the binomial start system — every start root
            // is an exact root of H(·, 0).
            for start in start_solutions(&system, cell) {
                let h0 = hom.eval(&start, 0.0);
                assert!(h0[0].magnitude() < 1e-9 && h0[1].magnitude() < 1e-9);
            }

            // Mid-path H agrees with a direct term-by-term evaluation
            // c·t^e·e^{iγe(1−t)}·y^a done independently of MSystem,
            // including the t ↦ t^(1/e_min) level normalization and the
            // γ-twist phase.
            let raw_level = |i: usize, m: &Monomial<2>| -> f64 {
                let level = |mm: &Monomial<2>| -> f64 {
                    let j = supports[i].points().iter().position(|p| p == mm).unwrap();
                    liftings[i].values()[j]
                        + mm.exps[0] as f64 * cell.normal[0]
                        + mm.exps[1] as f64 * cell.normal[1]
                };
                let (a, b) = cell.edges[i];
                if *m == a || *m == b {
                    0.0
                } else {
                    level(m) - level(&a).min(level(&b))
                }
            };
            let mut e_min = f64::INFINITY;
            for (i, poly) in system.polys.iter().enumerate() {
                for m in poly.support.iter() {
                    let e = raw_level(i, m);
                    if e > 0.0 && e < e_min {
                        e_min = e;
                    }
                }
            }
            let y = [c64::new(0.3, -0.8), c64::new(-1.1, 0.4)];
            let t = 0.37;
            let sys_t = hom.system_at(t);
            assert_eq!(hom.gamma(), 2f64.ln()); // the documented default
            for (i, poly) in system.polys.iter().enumerate() {
                let mut acc = c64::new(0.0, 0.0);
                for (c, m) in poly.coeffs.iter().zip(poly.support.iter()) {
                    let mut mono = c64::new(1.0, 0.0);
                    for (&yj, &e) in y.iter().zip(m.exps.iter()) {
                        mono *= yj.powi(e);
                    }
                    let level = raw_level(i, m) / e_min;
                    let twist = c64::from_polar(1.0, hom.gamma() * level * (1.0 - t));
                    acc += *c * mono * t.powf(level) * twist;
                }
                let got = sys_t.eval(&y)[i];
                assert!(
                    (got - acc).magnitude() < 1e-12,
                    "H_{} mismatch: {} vs {}",
                    i,
                    got,
                    acc
                );
            }

            // γ = 0 recovers the untwisted textbook homotopy c·t^e·y^a.
            let plain = CellHomotopy::with_gamma(&system, &supports, &liftings, cell, 0.0);
            assert_eq!(plain.gamma(), 0.0);
            let sys_plain = plain.system_at(t);
            for (i, poly) in system.polys.iter().enumerate() {
                let mut acc = c64::new(0.0, 0.0);
                for (c, m) in poly.coeffs.iter().zip(poly.support.iter()) {
                    let mut mono = c64::new(1.0, 0.0);
                    for (&yj, &e) in y.iter().zip(m.exps.iter()) {
                        mono *= yj.powi(e);
                    }
                    acc += *c * mono * t.powf(raw_level(i, m) / e_min);
                }
                let got = sys_plain.eval(&y)[i];
                assert!(
                    (got - acc).magnitude() < 1e-12,
                    "untwisted H_{} mismatch: {} vs {}",
                    i,
                    got,
                    acc
                );
            }
        }
    }

    #[test]
    fn dt_matches_finite_differences() {
        let system = trinomial();
        let supports = Support::from_msystem(&system);
        let liftings = random_liftings::<f64, 2>(&supports, 2026);
        let cells = mixed_cells(&supports, &liftings).unwrap();
        let hom = CellHomotopy::new(&system, &supports, &liftings, &cells[0]);

        let y = [c64::new(0.7, 0.2), c64::new(-0.5, 1.3)];
        for t in [0.2, 0.5, 0.8] {
            let d = hom.dt(&y, t);
            let h = 1e-6;
            let hp = hom.eval(&y, t + h);
            let hm = hom.eval(&y, t - h);
            for k in 0..2 {
                let fd = (hp[k] - hm[k]) / (2.0 * h);
                assert!(
                    (d[k] - fd).magnitude() < 1e-6,
                    "dt vs finite difference at t = {}: {} vs {}",
                    t,
                    d[k],
                    fd
                );
            }
        }
    }

    #[test]
    fn track_paths_of_the_trinomial_pair() {
        let system = trinomial();
        let supports = Support::from_msystem(&system);
        let liftings = random_liftings::<f64, 2>(&supports, 2026);
        let cells = mixed_cells(&supports, &liftings).unwrap();
        let options = TrackOptions::default();

        let mut endpoints: Vec<[c64; 2]> = Vec::new();
        for cell in &cells {
            let hom = CellHomotopy::new(&system, &supports, &liftings, cell);
            for start in start_solutions(&system, cell) {
                let path = track_path(&hom, start, &options);
                assert_eq!(path.status, PathStatus::Converged);
                assert_eq!(path.t_reached, 1.0);
                assert!(path.steps > 0 && path.newton_iters > 0);
                assert!(
                    residual_inf(&system, &path.point) < 1e-9,
                    "endpoint is not a root of the target"
                );
                endpoints.push(path.point);
            }
        }
        // Mixed volume 2: two paths, two distinct roots.
        assert_eq!(endpoints.len(), 2);
        let d = (endpoints[0][0] - endpoints[1][0])
            .magnitude()
            .max((endpoints[0][1] - endpoints[1][1]).magnitude());
        assert!(d > 1e-6, "paths collapsed to one root");
    }

    #[test]
    fn budget_statuses_are_honest() {
        let system = trinomial();
        let supports = Support::from_msystem(&system);
        let liftings = random_liftings::<f64, 2>(&supports, 2026);
        let cells = mixed_cells(&supports, &liftings).unwrap();
        let hom = CellHomotopy::new(&system, &supports, &liftings, &cells[0]);
        let start = start_solutions(&system, &cells[0])[0];

        // One attempted step can't reach t = 1.
        let opts = TrackOptions {
            max_steps: 1,
            ..TrackOptions::default()
        };
        let path = track_path(&hom, start, &opts);
        assert_eq!(path.status, PathStatus::MaxStepsReached);
        assert!(path.t_reached < 1.0);
        assert_eq!(path.steps, 1);

        // A corrector that may never iterate can never accept: dt halves to
        // the floor and the path reports MinStepReached at t = 0.
        let opts = TrackOptions {
            max_newton: 0,
            ..TrackOptions::default()
        };
        let path = track_path(&hom, start, &opts);
        assert_eq!(path.status, PathStatus::MinStepReached);
        assert_eq!(path.t_reached, 0.0);
        assert_eq!(path.newton_iters, 0);
    }

    #[test]
    fn path_status_displays_short_lowercase() {
        let cases = [
            (PathStatus::Converged, "converged"),
            (PathStatus::MinStepReached, "min-step"),
            (PathStatus::MaxStepsReached, "max-steps"),
            (PathStatus::SingularJacobian, "singular"),
            (PathStatus::Diverged, "diverged"),
        ];
        for (status, want) in cases {
            assert_eq!(format!("{}", status), want);
            // Width/alignment flags are honored (f.pad, not write_str).
            assert_eq!(format!("{:<10}|", status), format!("{:<10}|", want));
        }
    }

    #[test]
    fn default_options_match_spec() {
        let o = TrackOptions::<f64>::default();
        assert_eq!(o.predictor, Predictor::Rk4);
        assert_eq!(o.predictor, Predictor::default());
        assert!((o.dt_init - 1e-2).abs() < 1e-16);
        assert!((o.dt_min - 1e-14).abs() < 1e-28);
        assert!((o.dt_max - 1e-1).abs() < 1e-16);
        assert!((o.newton_tol - 1e-10).abs() < 1e-24);
        assert_eq!(o.max_newton, 3);
        assert_eq!(o.max_steps, 10_000);
        assert!((o.divergence_bound - 1e8).abs() < 1.0);
        assert_eq!(o.grow, 2.0);
        assert_eq!(o.shrink, 0.5);
    }
}
