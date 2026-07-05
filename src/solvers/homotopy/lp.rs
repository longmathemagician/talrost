//! A small dense feasibility LP — the pruning engine of the DEMiCs-style
//! mixed-cell tree search in [`super::cells`].
//!
//! [`max_delta`] solves, over `α ∈ ℝ^NV` and a scalar slack `δ`:
//!
//! ```text
//! maximize δ   subject to   A_eq·α = b_eq,   A_in·α − δ·1 ≥ b_in
//! ```
//!
//! The optimum `δ*` is the largest *uniform margin* by which the strict
//! system `{A_eq·α = b_eq, A_in·α > b_in}` can hold: `δ* > 0` certifies
//! strict feasibility (with a witness margin), `δ* ≤ 0` certifies that no
//! `α` satisfies every inequality strictly, and a `δ*` inside the caller's
//! genericity band around zero means the lifting is too close to a tie to
//! decide — the caller maps that to
//! [`super::cells::GenericityError`]. `δ` may also be unbounded above
//! (common at shallow tree depths, where the equalities leave `α` degrees
//! of freedom that increase every inequality slack simultaneously); that is
//! reported as [`DeltaMax::Unbounded`] and counts as strict feasibility.
//!
//! # Method
//!
//! A textbook dense two-phase primal simplex on the standard-form
//! translation (free variables `α_j` and `δ` split into non-negative pairs,
//! one surplus variable per inequality, one artificial variable per row):
//! phase 1 minimizes the artificial sum to find a basic feasible solution
//! (or proves the equalities inconsistent), phase 2 maximizes `δ`.
//! **Bland's smallest-index rule** picks both the entering column and the
//! leaving row (on ratio ties), which excludes cycling in exact arithmetic.
//! The problems this module sees are tiny — `NV + 1 ≤ 9` genuine unknowns
//! and a few dozen constraints — so simplicity is worth more than sparse
//! bases, steepest-edge pricing, or warm starts.
//!
//! # Tolerances
//!
//! All entries are exact small integers (exponent differences, `|entry| ≲
//! 2⁶`) and unit-interval lift differences, so the LP is well scaled and
//! absolute thresholds tied to the machine epsilon suffice:
//!
//! - [`pivot_eps`] (`1024·ε`, ≈ 2.3e-13 for `f64`): reduced costs and
//!   pivot candidates below this magnitude are treated as zero.
//! - [`feas_eps`] (`16384·ε`, ≈ 3.6e-12 for `f64`): the phase-1 artificial
//!   sum must fall below this for the constraints to count as consistent.
//!
//! Both sit far below the `1e-9` genericity band of [`super::cells`], so
//! simplex roundoff is absorbed by the band: an optimum contaminated by
//! ~1e-13 noise can only misclassify a system whose true margin is within
//! the band — exactly the case the caller already treats as "re-lift and
//! retry".
//!
//! # Bland's rule under floating point
//!
//! Bland's anti-cycling guarantee is a statement about exact arithmetic;
//! with rounded pivots and thresholded comparisons a cycle is not
//! *provably* impossible. Every simplex phase therefore carries a generous
//! iteration cap (`200 + 50·(rows + columns)`), and hitting it returns the
//! distinguished [`DeltaMax::Stalled`] instead of an unreliable optimum —
//! the caller treats a stall like a genericity failure (re-lift), never as
//! a pruning decision.

use crate::real::Real;

/// The outcome of the maximize-`δ` LP (see the module docs).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum DeltaMax<F> {
    /// No `α` satisfies the constraints at all (with `δ → −∞`): the
    /// equality system is inconsistent.
    Infeasible,
    /// The optimum `δ*`: the largest uniform slack of the inequalities over
    /// the equality-feasible set.
    Bounded(F),
    /// `δ` can be made arbitrarily large: strictly feasible with unbounded
    /// margin.
    Unbounded,
    /// An iteration cap was hit before optimality could be certified (see
    /// the module docs on Bland's rule under floating point); the result
    /// must be treated as "cannot decide", never as a pruning verdict.
    Stalled,
}

/// Zero threshold for reduced costs and pivot candidates: `1024·ε`.
fn pivot_eps<F: Real>() -> F {
    F::EPSILON * F::from_u32(1 << 10)
}

/// Phase-1 consistency threshold on the artificial sum: `16384·ε`.
fn feas_eps<F: Real>() -> F {
    F::EPSILON * F::from_u32(1 << 14)
}

/// One simplex phase's exit condition.
enum Step {
    /// No entering column improves the objective: current basis is optimal.
    Optimal,
    /// An entering column has no blocking row: the objective is unbounded.
    Unbounded,
    /// The iteration cap was hit.
    Stalled,
}

/// The dense phase state: `m` constraint rows over `n_struct` structural
/// columns (artificial columns are never materialized — an artificial basic
/// variable of row `r` is encoded as the sentinel basis index
/// `n_struct + r`, and artificials never re-enter the basis, so their
/// tableau columns are never read).
struct Tableau<F> {
    /// `m × n_struct` constraint coefficients, kept in canonical (basis =
    /// identity) form by pivoting.
    rows: Vec<Vec<F>>,
    /// Right-hand sides, non-negative up to roundoff dust.
    rhs: Vec<F>,
    /// `basis[r]` = the column basic in row `r` (or `n_struct + r` for the
    /// row's artificial).
    basis: Vec<usize>,
    /// Reduced costs of the structural columns for the current objective.
    reduced: Vec<F>,
    /// Current objective value.
    objective: F,
}

impl<F: Real> Tableau<F> {
    /// Pivots column `col` into the basis at row `row`, restoring canonical
    /// form and updating the reduced-cost row and objective value.
    fn pivot(&mut self, row: usize, col: usize) {
        let inv = F::ONE / self.rows[row][col];
        for v in self.rows[row].iter_mut() {
            *v *= inv;
        }
        self.rows[row][col] = F::ONE; // exact, not round-off
        self.rhs[row] *= inv;

        // Small dense problems: cloning the pivot row sidesteps the split
        // borrow and costs nothing that matters here.
        let prow = self.rows[row].clone();
        let prhs = self.rhs[row];
        for r in 0..self.rows.len() {
            if r == row {
                continue;
            }
            let f = self.rows[r][col];
            if f == F::ZERO {
                continue;
            }
            for (dst, src) in self.rows[r].iter_mut().zip(prow.iter()) {
                *dst -= f * *src;
            }
            self.rows[r][col] = F::ZERO; // exact
            self.rhs[r] -= f * prhs;
        }
        let f = self.reduced[col];
        if f != F::ZERO {
            for (dst, src) in self.reduced.iter_mut().zip(prow.iter()) {
                *dst -= f * *src;
            }
            self.reduced[col] = F::ZERO;
            self.objective += f * prhs;
        }
        self.basis[row] = col;
    }

    /// Runs simplex iterations under **Bland's rule** until optimality,
    /// unboundedness, or the iteration cap: the entering column is the
    /// smallest structural index with a positive reduced cost, the leaving
    /// row is the ratio-test minimum with ties broken by the smallest basic
    /// index. Artificials never enter (they have no reduced-cost entry).
    fn run(&mut self, eps: F) -> Step {
        let max_iters = 200 + 50 * (self.rows.len() + self.reduced.len());
        for _ in 0..max_iters {
            let Some(col) = self.reduced.iter().position(|d| *d > eps) else {
                return Step::Optimal;
            };
            // Ratio test: smallest rhs/coefficient over positive
            // coefficients; negative rhs dust counts as zero.
            let mut leave: Option<(usize, F)> = None;
            for (r, (row, &b)) in self.rows.iter().zip(self.rhs.iter()).enumerate() {
                let a = row[col];
                if a > eps {
                    let ratio = b.max(F::ZERO) / a;
                    let better = match leave {
                        None => true,
                        Some((lr, lratio)) => {
                            ratio < lratio || (ratio == lratio && self.basis[r] < self.basis[lr])
                        }
                    };
                    if better {
                        leave = Some((r, ratio));
                    }
                }
            }
            let Some((row, _)) = leave else {
                return Step::Unbounded;
            };
            self.pivot(row, col);
        }
        Step::Stalled
    }
}

/// Maximizes `δ` subject to `eq·α = rhs` for every `(eq, rhs)` in
/// `equalities` and `row·α − δ ≥ rhs` for every `(row, rhs)` in
/// `inequalities` (see the module docs for the method and tolerances).
///
/// With no constraints at all — or with consistent equalities and no
/// inequalities — `δ` is unconstrained and the result is
/// [`DeltaMax::Unbounded`].
pub(super) fn max_delta<F: Real, const NV: usize>(
    equalities: &[([F; NV], F)],
    inequalities: &[([F; NV], F)],
) -> DeltaMax<F> {
    let n_eq = equalities.len();
    let n_in = inequalities.len();
    let m = n_eq + n_in;
    if m == 0 {
        return DeltaMax::Unbounded;
    }

    // Standard-form columns: α⁺/α⁻ pairs, δ⁺/δ⁻, one surplus per
    // inequality. Column j of α_j maps to 2j (positive part) and 2j+1
    // (negative part); δ maps to 2·NV and 2·NV+1.
    let n_struct = 2 * NV + 2 + n_in;
    let d_pos = 2 * NV;
    let d_neg = 2 * NV + 1;

    let mut rows: Vec<Vec<F>> = Vec::with_capacity(m);
    let mut rhs: Vec<F> = Vec::with_capacity(m);
    for (r, (coeffs, b)) in equalities.iter().chain(inequalities.iter()).enumerate() {
        let mut row = vec![F::ZERO; n_struct];
        for (j, &c) in coeffs.iter().enumerate() {
            row[2 * j] = c;
            row[2 * j + 1] = -c;
        }
        let mut b = *b;
        if r >= n_eq {
            // Inequality r − n_eq: coeffs·α − δ − s = b, s ≥ 0.
            row[d_pos] = -F::ONE;
            row[d_neg] = F::ONE;
            row[2 * NV + 2 + (r - n_eq)] = -F::ONE;
        }
        if b < F::ZERO {
            for v in row.iter_mut() {
                *v = -*v;
            }
            b = -b;
        }
        rows.push(row);
        rhs.push(b);
    }

    let eps = pivot_eps::<F>();

    // Phase 1: maximize −Σ artificials from the all-artificial basis. With
    // basis = artificials (cost −1 each), the reduced cost of structural
    // column j is Σ_r rows[r][j] and the objective starts at −Σ rhs.
    let mut reduced = vec![F::ZERO; n_struct];
    for (j, d) in reduced.iter_mut().enumerate() {
        let mut acc = F::ZERO;
        for row in &rows {
            acc += row[j];
        }
        *d = acc;
    }
    let mut objective = F::ZERO;
    for &b in &rhs {
        objective -= b;
    }
    let mut t = Tableau {
        rows,
        rhs,
        basis: (n_struct..n_struct + m).collect(),
        reduced,
        objective,
    };
    match t.run(eps) {
        Step::Optimal => {}
        // Phase 1's objective is ≤ 0 by construction; "unbounded" can only
        // be numerical debris, so refuse to certify anything.
        Step::Unbounded | Step::Stalled => return DeltaMax::Stalled,
    }
    if t.objective < -feas_eps::<F>() {
        return DeltaMax::Infeasible;
    }

    // Drive surviving artificials out of the (degenerate) basis so phase-2
    // pivots cannot revive them. A row whose structural coefficients are
    // all ~0 is a redundant constraint; its artificial stays basic at zero
    // and is inert (every entering column has a ~0 coefficient there).
    for r in 0..m {
        if t.basis[r] >= n_struct {
            if let Some(col) = (0..n_struct).find(|&j| t.rows[r][j].abs() > eps) {
                t.pivot(r, col);
            }
        }
    }

    // Phase 2: maximize δ = δ⁺ − δ⁻. Rebuild the reduced costs for the new
    // objective from the current basis (artificial basics cost zero).
    let cost = |j: usize| -> F {
        if j == d_pos {
            F::ONE
        } else if j == d_neg {
            -F::ONE
        } else {
            F::ZERO
        }
    };
    for j in 0..n_struct {
        let mut acc = cost(j);
        for (row, &b) in t.rows.iter().zip(t.basis.iter()) {
            if b < n_struct {
                acc -= cost(b) * row[j];
            }
        }
        t.reduced[j] = acc;
    }
    t.objective = F::ZERO;
    for (&b, &v) in t.basis.iter().zip(t.rhs.iter()) {
        if b < n_struct {
            t.objective += cost(b) * v;
        }
    }
    // Basic columns must read as reduced cost zero exactly (canonical
    // form); enforce it against summation dust.
    for &b in &t.basis {
        if b < n_struct {
            t.reduced[b] = F::ZERO;
        }
    }

    match t.run(eps) {
        Step::Optimal => DeltaMax::Bounded(t.objective),
        Step::Unbounded => DeltaMax::Unbounded,
        Step::Stalled => DeltaMax::Stalled,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::Matrix;
    use crate::vector::Vector;

    /// Strictly feasible 2-D system: α inside the unit square around
    /// (0.5, 0.5), no equalities. δ* is the largest inscribed margin.
    #[test]
    fn strictly_feasible_box() {
        // α₀ ≥ 0, α₁ ≥ 0, −α₀ ≥ −1, −α₁ ≥ −1.
        let ins: [([f64; 2], f64); 4] = [
            ([1.0, 0.0], 0.0),
            ([0.0, 1.0], 0.0),
            ([-1.0, 0.0], -1.0),
            ([0.0, -1.0], -1.0),
        ];
        match max_delta::<f64, 2>(&[], &ins) {
            DeltaMax::Bounded(d) => assert!((d - 0.5).abs() < 1e-12, "δ* = {}", d),
            other => panic!("expected Bounded, got {:?}", other),
        }
    }

    /// Infeasible 2-D system: α₀ ≥ 1 and −α₀ ≥ 0 cannot both hold
    /// strictly; the best uniform margin is negative.
    #[test]
    fn infeasible_halfplanes() {
        let ins: [([f64; 2], f64); 2] = [([1.0, 0.0], 1.0), ([-1.0, 0.0], 0.0)];
        match max_delta::<f64, 2>(&[], &ins) {
            // max δ s.t. α₀ − δ ≥ 1, −α₀ − δ ≥ 0 ⇒ δ* = −1/2.
            DeltaMax::Bounded(d) => assert!((d + 0.5).abs() < 1e-12, "δ* = {}", d),
            other => panic!("expected Bounded, got {:?}", other),
        }
    }

    /// A degenerate tie: three halfplanes meeting at one point. The strict
    /// system is infeasible but the weak one holds exactly at the vertex,
    /// so δ* = 0 (the ambiguous band's center).
    #[test]
    fn degenerate_tie_is_delta_zero() {
        // α₀ ≥ 0, α₁ ≥ 0, −α₀ − α₁ ≥ 0: only α = (0, 0) weakly.
        let ins: [([f64; 2], f64); 3] = [([1.0, 0.0], 0.0), ([0.0, 1.0], 0.0), ([-1.0, -1.0], 0.0)];
        match max_delta::<f64, 2>(&[], &ins) {
            DeltaMax::Bounded(d) => assert!(d.abs() < 1e-12, "δ* = {}", d),
            other => panic!("expected Bounded, got {:?}", other),
        }
    }

    /// Unboundedness: a single lower halfplane leaves δ free to grow with
    /// α₀. And with no constraints at all, δ is trivially unbounded.
    #[test]
    fn unbounded_margin() {
        let ins: [([f64; 2], f64); 1] = [([1.0, 0.0], 0.0)];
        assert_eq!(max_delta::<f64, 2>(&[], &ins), DeltaMax::Unbounded);
        assert_eq!(max_delta::<f64, 2>(&[], &[]), DeltaMax::Unbounded);
    }

    /// Equality-only systems, cross-checked against the LU solve: a square
    /// nonsingular equality system is always consistent (δ unconstrained →
    /// unbounded), and pinning α by equalities makes any inequality's
    /// margin computable directly from the LU solution.
    #[test]
    fn equalities_cross_checked_against_lu() {
        let a = [[2.0, 1.0, 0.0], [1.0, -1.0, 3.0], [0.0, 2.0, -1.0]];
        let b = [0.7, -0.3, 1.9];
        let eqs: [([f64; 3], f64); 3] = [(a[0], b[0]), (a[1], b[1]), (a[2], b[2])];

        // Consistent equalities, no inequalities: unbounded.
        assert_eq!(max_delta::<f64, 3>(&eqs, &[]), DeltaMax::Unbounded);

        // The unique α from the LU solve fixes every inequality margin:
        // for row (c, r), δ* = c·α − r.
        let alpha = Matrix::new(a).solve(&Vector::new(b)).unwrap();
        let c = [0.5, -2.0, 1.5];
        let r = -0.9;
        let expect = c[0] * alpha[0] + c[1] * alpha[1] + c[2] * alpha[2] - r;
        match max_delta::<f64, 3>(&eqs, &[(c, r)]) {
            DeltaMax::Bounded(d) => {
                assert!((d - expect).abs() < 1e-12, "δ* = {} vs LU {}", d, expect)
            }
            other => panic!("expected Bounded, got {:?}", other),
        }

        // An inconsistent equality pair is infeasible outright.
        let bad: [([f64; 3], f64); 2] = [([1.0, 1.0, 0.0], 1.0), ([2.0, 2.0, 0.0], 3.0)];
        assert_eq!(max_delta::<f64, 3>(&bad, &[]), DeltaMax::Infeasible);

        // Redundant-but-consistent equalities are fine (rank-deficient
        // phase 1 leaves an inert artificial basic at zero).
        let dup: [([f64; 3], f64); 2] = [([1.0, 1.0, 0.0], 1.0), ([2.0, 2.0, 0.0], 2.0)];
        match max_delta::<f64, 3>(&dup, &[(c, r)]) {
            DeltaMax::Bounded(_) | DeltaMax::Unbounded => {}
            other => panic!("expected feasible, got {:?}", other),
        }
    }

    /// Mixed equalities and inequalities with a hand-checkable optimum:
    /// on the line α₀ + α₁ = 1, maximize min(α₀, α₁) — the balanced point
    /// (1/2, 1/2) gives δ* = 1/2.
    #[test]
    fn equality_restricted_margin() {
        let eqs: [([f64; 2], f64); 1] = [([1.0, 1.0], 1.0)];
        let ins: [([f64; 2], f64); 2] = [([1.0, 0.0], 0.0), ([0.0, 1.0], 0.0)];
        match max_delta::<f64, 2>(&eqs, &ins) {
            DeltaMax::Bounded(d) => assert!((d - 0.5).abs() < 1e-12, "δ* = {}", d),
            other => panic!("expected Bounded, got {:?}", other),
        }
    }
}
