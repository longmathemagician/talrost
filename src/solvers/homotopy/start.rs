//! Binomial start systems of mixed cells, solved in closed form via the
//! Smith normal form.
//!
//! For a target system `F` (complex coefficients, `NEQ == NV`) and a fine
//! mixed cell with edges `(a_i, b_i)`, the start system keeps only the two
//! level terms of each equation: `c_{a_i}·x^{a_i} + c_{b_i}·x^{b_i} = 0`,
//! equivalently **`x^V = β`** with `V` the cell's edge matrix (rows
//! `v_i = b_i − a_i`) and `β_i = −c_{a_i}/c_{b_i} ∈ ℂ*`.
//!
//! # Conventions (they matter)
//!
//! Matrix powers of points follow `(x^M)_i = Π_j x_j^{M_ij}` — row `i` is
//! equation `i` — so substitution composes as `x = z^A ⇒ x^B = z^{B·A}`.
//! With `(U, S, W) = smith_normal_form(V)`, i.e. `U·V·W = S = diag(s_i)`
//! with `U`, `W` unimodular:
//!
//! - raising the system to `U` gives `x^{U·V} = β^U` with
//!   `(β^U)_i = Π_j β_j^{U_ij}` — solution-preserving on the torus because
//!   `U` is invertible over ℤ;
//! - substituting `x = y^W` (a torus bijection, `W` unimodular) turns the
//!   left side into `(y^W)^{U·V} = y^{U·V·W} = y^S`;
//! - the system is now diagonal, **`y_i^{s_i} = γ_i`** with `γ = β^U`, so
//!   `y_i` ranges over the `s_i` distinct `s_i`-th roots of `γ_i`
//!   ([`Complex::nth_roots`]) and there are `Π_i s_i = |det V|` solutions;
//! - mapping back through `x = y^W` yields the start points, distinct
//!   because `y ↦ y^W` is injective on the torus.

use crate::algebra::Field;
use crate::complex::Complex;
use crate::lattice::smith_normal_form;
use crate::matrix::Matrix;
use crate::mvpoly::{MPoly, MSystem, Monomial};
use crate::real::Real;
use crate::scalar::Scalar;

use super::cells::MixedCell;

/// `z^n` for an `i64` exponent by exponentiation-by-squaring; negative
/// exponents go through [`Field::recip`] first (requiring `z ≠ 0`, exactly
/// like [`Complex::powi`] — which this widens, since SNF transform matrices
/// can in principle carry entries outside `i32`).
fn powi64<F: Real>(z: Complex<F>, n: i64) -> Complex<F> {
    let mut base = if n < 0 { z.recip() } else { z };
    let mut exp = n.unsigned_abs();
    let mut acc = Complex::new(F::ONE, F::ZERO);
    while exp > 0 {
        if exp & 1 == 1 {
            acc *= base;
        }
        base *= base;
        exp >>= 1;
    }
    acc
}

/// All solutions in `(ℂ*)^NV` of the binomial system `x^V = β`, i.e.
/// `Π_j x_j^{V_ij} = β_i` for every `i`, via the Smith normal form of `V`
/// (see the module docs for the exact derivation and conventions).
///
/// Returns exactly `|det V|` pairwise-distinct points, in the deterministic
/// order induced by enumerating the per-coordinate root indices
/// `k_i ∈ 0..s_i` odometer-style (first coordinate fastest).
///
/// # Panics
///
/// Panics if `V` is singular (some `s_i = 0`) — the system then has no
/// isolated torus solutions; a genuine mixed cell's edge matrix never is.
/// Every `β_i` must be nonzero (`debug_assert`ed): zeros are outside the
/// torus and would produce non-finite garbage.
pub fn binomial_solutions<F: Real, const NV: usize>(
    v: &Matrix<i64, NV, NV>,
    beta: &[Complex<F>; NV],
) -> Vec<[Complex<F>; NV]> {
    for b in beta.iter() {
        debug_assert!(
            b.norm_sqr() > F::ZERO,
            "binomial_solutions: β must lie in the torus (all entries nonzero)"
        );
    }
    let (u, s, w) = smith_normal_form(v);

    // γ = β^U: γ_i = Π_j β_j^{U_ij}.
    let mut gamma = [Complex::new(F::ONE, F::ZERO); NV];
    for (gi, urow) in gamma.iter_mut().zip(u.e.iter()) {
        for (&bj, &uij) in beta.iter().zip(urow.iter()) {
            if uij != 0 {
                *gi *= powi64(bj, uij);
            }
        }
    }

    // Invariant factors: s_i > 0 exactly when V is nonsingular.
    let mut counts = [0u32; NV];
    let mut total = 1usize;
    for (i, c) in counts.iter_mut().enumerate() {
        let si = s.e[i][i];
        assert!(
            si > 0,
            "binomial_solutions: singular exponent matrix (det V = 0)"
        );
        *c = u32::try_from(si).expect("invariant factor exceeds u32");
        total *= *c as usize;
    }

    // Per-coordinate root menus: y_i is one of the s_i-th roots of γ_i.
    let roots: [Vec<Complex<F>>; NV] =
        core::array::from_fn(|i| gamma[i].nth_roots(counts[i]).collect());

    // Every combination, mapped back through x = y^W: x_i = Π_j y_j^{W_ij}.
    let mut out = Vec::with_capacity(total);
    let mut k = [0usize; NV];
    loop {
        let mut x = [Complex::new(F::ONE, F::ZERO); NV];
        for (xi, wrow) in x.iter_mut().zip(w.e.iter()) {
            let mut acc = Complex::new(F::ONE, F::ZERO);
            for ((rj, &kj), &wij) in roots.iter().zip(k.iter()).zip(wrow.iter()) {
                if wij != 0 {
                    acc *= powi64(rj[kj], wij);
                }
            }
            *xi = acc;
        }
        out.push(x);
        if NV == 0 {
            return out; // the empty system has the single empty solution
        }

        // Advance the root-index odometer.
        let mut d = 0;
        loop {
            k[d] += 1;
            if k[d] < counts[d] as usize {
                break;
            }
            k[d] = 0;
            d += 1;
            if d == NV {
                return out;
            }
        }
    }
}

/// The total coefficient of monomial `m` in `poly`: repeated monomials are
/// summed (zero-coefficient padding terms contribute nothing either way).
fn coefficient_of<F: Real, const NV: usize, const MAXT: usize>(
    poly: &MPoly<Complex<F>, NV, MAXT>,
    m: &Monomial<NV>,
) -> Complex<F> {
    let mut acc = Complex::new(F::ZERO, F::ZERO);
    for (c, mm) in poly.coeffs.iter().zip(poly.support.iter()) {
        if mm == m {
            acc += *c;
        }
    }
    acc
}

/// The start solutions of `cell` for the target `system`: the roots of the
/// cell's binomial start system `c_{a_i}·x^{a_i} + c_{b_i}·x^{b_i} = 0`.
///
/// Builds `β_i = −c_{a_i}/c_{b_i}` from the coefficients of the cell's edge
/// monomials in equation `i` and hands `x^V = β` to
/// [`binomial_solutions`]; the result has exactly [`MixedCell::volume`]
/// entries. These are the `t = 0` endpoints the (future) path tracker will
/// continue toward the roots of `system`.
///
/// # Panics
///
/// Panics if an edge coefficient of some equation vanishes — the cell was
/// then computed for a support this system does not actually have (its
/// binomial subsystem degenerates to a monomial with no torus roots).
pub fn start_solutions<F: Real, const NV: usize, const MAXT: usize>(
    system: &MSystem<Complex<F>, NV, NV, MAXT>,
    cell: &MixedCell<F, NV>,
) -> Vec<[Complex<F>; NV]> {
    let mut beta = [Complex::new(F::ZERO, F::ZERO); NV];
    for ((bi, poly), (a, b)) in beta
        .iter_mut()
        .zip(system.polys.iter())
        .zip(cell.edges.iter())
    {
        let ca = coefficient_of(poly, a);
        let cb = coefficient_of(poly, b);
        assert!(
            ca.norm_sqr() > F::ZERO && cb.norm_sqr() > F::ZERO,
            "start_solutions: an edge monomial has zero coefficient in its equation"
        );
        *bi = -(ca / cb);
    }
    binomial_solutions(&cell.edge_matrix, &beta)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::complex::c64;
    use crate::solvers::homotopy::support::{random_liftings, Support};
    use crate::solvers::homotopy::{mixed_cells, mixed_volume};

    /// Residual `|x^v − β|` of one binomial equation at a point.
    fn binomial_residual(x: &[c64], vrow: &[i64], beta: c64) -> f64 {
        let mut acc = c64::new(1.0, 0.0);
        for (&xj, &vj) in x.iter().zip(vrow.iter()) {
            acc *= powi64(xj, vj);
        }
        (acc - beta).magnitude()
    }

    /// Asserts `sols` and `expected` agree as sets (within `tol`) and that
    /// `sols` are pairwise distinct.
    fn assert_solution_set(sols: &[[c64; 2]], expected: &[[c64; 2]], tol: f64) {
        assert_eq!(sols.len(), expected.len());
        for e in expected {
            assert!(
                sols.iter()
                    .any(|s| (s[0] - e[0]).magnitude() < tol && (s[1] - e[1]).magnitude() < tol),
                "expected solution {:?} not found",
                e
            );
        }
        for (i, s) in sols.iter().enumerate() {
            for t in sols.iter().skip(i + 1) {
                let d = (s[0] - t[0]).magnitude() + (s[1] - t[1]).magnitude();
                assert!(d > 1e-8, "duplicate solutions {:?}", s);
            }
        }
    }

    #[test]
    fn univariate_square_root_of_unity() {
        // x² = 1 through the full SNF machinery: solutions ±1.
        let v = Matrix::<i64, 1, 1>::new([[2]]);
        let sols = binomial_solutions(&v, &[c64::new(1.0, 0.0)]);
        assert_eq!(sols.len(), 2);
        let mut found_pos = false;
        let mut found_neg = false;
        for s in &sols {
            if (s[0] - c64::new(1.0, 0.0)).magnitude() < 1e-12 {
                found_pos = true;
            }
            if (s[0] - c64::new(-1.0, 0.0)).magnitude() < 1e-12 {
                found_neg = true;
            }
        }
        assert!(found_pos && found_neg);
    }

    #[test]
    fn diagonal_system_exact_roots() {
        // x₁² = 4, x₂³ = 8: 6 solutions, each verified exactly. (Note the
        // SNF of diag(2, 3) is diag(1, 6) — the machinery must still emit
        // the correct product set.)
        let v = Matrix::<i64, 2, 2>::new([[2, 0], [0, 3]]);
        let beta = [c64::new(4.0, 0.0), c64::new(8.0, 0.0)];
        let sols = binomial_solutions(&v, &beta);

        let omega = c64::nth_root_of_unity(1, 3);
        let x2s = [c64::new(2.0, 0.0), omega * 2.0, omega * omega * 2.0];
        let mut expected = Vec::new();
        for x1 in [c64::new(2.0, 0.0), c64::new(-2.0, 0.0)] {
            for x2 in x2s {
                expected.push([x1, x2]);
            }
        }
        assert_solution_set(&sols, &expected, 1e-10);
        for s in &sols {
            assert!(binomial_residual(s, &[2, 0], beta[0]) < 1e-10);
            assert!(binomial_residual(s, &[0, 3], beta[1]) < 1e-10);
        }
    }

    #[test]
    fn non_diagonal_edge_matrix() {
        // V = [[1, 1], [0, 2]] (det 2) with β of non-trivial argument:
        // x₁x₂ = β₁, x₂² = β₂. Every returned point must satisfy both
        // binomials to 1e-10, count must equal |det V|, and the solutions
        // must be pairwise distinct.
        let v = Matrix::<i64, 2, 2>::new([[1, 1], [0, 2]]);
        let beta = [c64::from_polar(2.0, 0.7), c64::from_polar(3.0, -1.3)];
        let sols = binomial_solutions(&v, &beta);
        assert_eq!(sols.len(), 2);
        for s in &sols {
            assert!(binomial_residual(s, &v.e[0], beta[0]) < 1e-10);
            assert!(binomial_residual(s, &v.e[1], beta[1]) < 1e-10);
        }
        let d = (sols[0][0] - sols[1][0]).magnitude() + (sols[0][1] - sols[1][1]).magnitude();
        assert!(d > 1e-8);

        // A denser V with negative entries and det 3.
        let v = Matrix::<i64, 2, 2>::new([[2, -1], [1, 1]]);
        let beta = [c64::from_polar(0.5, 2.1), c64::from_polar(4.0, 0.3)];
        let sols = binomial_solutions(&v, &beta);
        assert_eq!(sols.len(), 3);
        for s in &sols {
            assert!(binomial_residual(s, &v.e[0], beta[0]) < 1e-10);
            assert!(binomial_residual(s, &v.e[1], beta[1]) < 1e-10);
        }
    }

    #[test]
    fn end_to_end_sparse_pair() {
        // The mixed-volume-2 trinomial pair with (fixed) random complex
        // coefficients: lift → cells → starts. The start count must equal
        // the mixed volume, and every start must satisfy its cell's
        // binomial subsystem.
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
        let supports = Support::from_msystem(&system);

        let seed = 2026;
        let mv = mixed_volume::<f64, 2>(&supports, seed).unwrap();
        assert_eq!(mv, 2);

        let liftings = random_liftings::<f64, 2>(&supports, seed);
        let cells = mixed_cells(&supports, &liftings).unwrap();

        let mut total = 0u64;
        for cell in &cells {
            let starts = start_solutions(&system, cell);
            assert_eq!(starts.len() as u64, cell.volume());
            total += starts.len() as u64;

            // Each start satisfies the cell's binomial subsystem
            // c_a·x^a + c_b·x^b = 0 to 1e-9.
            for x in &starts {
                for (i, (a, b)) in cell.edges.iter().enumerate() {
                    let ca = coefficient_of(&system.polys[i], a);
                    let cb = coefficient_of(&system.polys[i], b);
                    let mono = |m: &Monomial<2>| -> c64 {
                        let mut acc = c64::new(1.0, 0.0);
                        for (&xj, &e) in x.iter().zip(m.exps.iter()) {
                            acc *= xj.powi(e);
                        }
                        acc
                    };
                    let residual = (ca * mono(a) + cb * mono(b)).magnitude();
                    assert!(residual < 1e-9, "binomial residual {} too large", residual);
                }
            }
        }
        assert_eq!(total, mv);
    }

    #[test]
    fn start_count_matches_cell_volume_for_conics() {
        // Dense conic pair: Σ_cells #starts == mixed volume == 4, with
        // small residuals throughout.
        let c = c64::new;
        let coeffs1 = [
            c(1.1, 0.3),
            c(-0.7, 0.9),
            c(0.5, -1.3),
            c(2.0, 0.1),
            c(-1.4, -0.8),
            c(0.6, 1.7),
        ];
        let coeffs2 = [
            c(-0.9, 1.2),
            c(1.8, -0.4),
            c(0.3, 0.7),
            c(-1.1, -1.6),
            c(0.8, 0.2),
            c(1.5, -0.5),
        ];
        let monos = [
            Monomial::new([0, 0]),
            Monomial::new([1, 0]),
            Monomial::new([0, 1]),
            Monomial::new([2, 0]),
            Monomial::new([1, 1]),
            Monomial::new([0, 2]),
        ];
        let system = MSystem::new([MPoly::new(coeffs1, monos), MPoly::new(coeffs2, monos)]);
        let supports = Support::from_msystem(&system);

        let seed = 7;
        let liftings = random_liftings::<f64, 2>(&supports, seed);
        let cells = mixed_cells(&supports, &liftings).unwrap();
        let total: u64 = cells
            .iter()
            .map(|cell| {
                let starts = start_solutions(&system, cell);
                assert_eq!(starts.len() as u64, cell.volume());
                starts.len() as u64
            })
            .sum();
        assert_eq!(total, 4);
        assert_eq!(mixed_volume::<f64, 2>(&supports, seed).unwrap(), 4);
    }

    #[test]
    #[should_panic(expected = "singular exponent matrix")]
    fn singular_edge_matrix_panics() {
        let v = Matrix::<i64, 2, 2>::new([[1, 1], [2, 2]]);
        let _ = binomial_solutions(&v, &[c64::new(1.0, 0.0), c64::new(1.0, 0.0)]);
    }
}
