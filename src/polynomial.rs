//! Dense univariate polynomials with compile-time degree.
//!
//! # Coefficient convention (ascending)
//!
//! `Polynomial<T, N>` stores `N` coefficients in **ascending-power order**:
//! `c[i]` is the coefficient of `x^i`, so the polynomial is
//! `c[0] + c[1]·x + … + c[N-1]·x^(N-1)`.
//!
//! The index of a coefficient names its power independently of `N` — the same
//! convention as the multivariate types in [`crate::mvpoly`] — which makes
//! derivative/deflation index arithmetic trivial (`d/dx` maps `c[i]` to
//! `i·c[i]` at index `i-1`). `Display` still prints highest-degree-first;
//! presentation is independent of storage.
//!
//! # Root finding
//!
//! `roots(tol)` is provided as per-size inherent impls for `N = 2..=5`
//! (linear through quartic). Each returns a [`Roots`] value with the
//! **contract**: roots are sorted ascending and `len()` is the number of real
//! roots found (a repeated root may appear once per multiplicity, depending on
//! the solver).
//!
//! There is deliberately no `roots` for `N = 1`: a constant polynomial has
//! either no roots (`c ≠ 0`) or all of ℝ (`c = 0`), and neither is
//! representable as a finite set of isolated roots.

use crate::algebra::{Algebra, Ring};
use crate::real::Real;
use crate::scalar::Scalar;
use crate::solvers;

pub use crate::roots::Roots;

/// A dense univariate polynomial `c[0] + c[1]·x + … + c[N-1]·x^(N-1)`.
///
/// **Coefficients are stored in ascending-power order**: `c[i]` multiplies
/// `x^i`. See the module docs for the rationale.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Polynomial<T, const N: usize> {
    /// The coefficients, **ascending**: `c[i]` multiplies `x^i`.
    pub c: [T; N],
}

impl<T, const N: usize> Polynomial<T, N> {
    /// Builds a polynomial from its ascending coefficient array (`c[i]`
    /// multiplies `x^i`).
    pub const fn new(c: [T; N]) -> Self {
        Self { c }
    }
}

impl<T, const N: usize> From<[T; N]> for Polynomial<T, N> {
    fn from(c: [T; N]) -> Self {
        Self::new(c)
    }
}

impl<T: Ring, const N: usize> Polynomial<T, N> {
    /// Evaluates the polynomial at a point of any [`Algebra`] over the
    /// coefficient ring `T` — one Horner body (plain mul+add) covering plain
    /// evaluation (`X = T`), real coefficients at complex points
    /// (Aberth/Durand–Kerner, the γ-trick), and derivatives via
    /// [`crate::dual::Dual`]/[`crate::dual::DualN`].
    ///
    /// The monomorphic [`Polynomial::eval`] is the fast path for `X = T =`
    /// a [`Scalar`]: same Horner recurrence, but accumulated with
    /// [`Scalar::mul_add_fast`]. Use `eval` when evaluating a scalar
    /// polynomial at a scalar point; use `eval_at` for everything else.
    pub fn eval_at<X: Algebra<T>>(&self, x: X) -> X {
        if N == 0 {
            return X::ZERO;
        }
        // Horner folds from the top coefficient down.
        self.c
            .iter()
            .rev()
            .skip(1)
            .fold(X::from(self.c[N - 1]), |acc, &k| acc * x + k)
    }
}

impl<T: Scalar, const N: usize> Polynomial<T, N> {
    /// Evaluates the polynomial at `x` by Horner's method, folding from the
    /// top coefficient `c[N-1]` down (monomorphization fully unrolls the fold
    /// for each `N`). The accumulation uses [`Scalar::mul_add_fast`]:
    /// hardware FMA where the target has it, plain multiply-add elsewhere (a
    /// fused `mul_add` would fall back to a libm software-fma call several
    /// times slower than the arithmetic itself).
    ///
    /// For evaluation at non-`T` points (complex points of a real polynomial,
    /// dual numbers, …) see [`Polynomial::eval_at`].
    ///
    /// A degenerate `Polynomial<T, 0>` has no coefficients and evaluates to
    /// zero (the empty sum).
    pub fn eval(&self, x: T) -> T {
        if N == 0 {
            return T::ZERO;
        }
        self.c
            .iter()
            .rev()
            .skip(1)
            .fold(self.c[N - 1], |acc, &k| acc.mul_add_fast(x, k))
    }
}

impl<T: Real, const N: usize> Polynomial<T, N> {
    /// Evaluates the polynomial at `x` together with a running bound on the
    /// evaluation's own rounding error (Higham, *Accuracy and Stability of
    /// Numerical Algorithms*, 2nd ed., Algorithm 5.1).
    ///
    /// Alongside the Horner value, it accumulates `μ_k = μ_{k-1}·|x| +
    /// |acc_k|` and returns `(value, bound)` with `bound = u·(2μ − |value|)`
    /// where `u` is half `EPSILON` (the unit roundoff); the true error
    /// `|computed − exact|` is bounded by `bound` to first order in `u`.
    ///
    /// This turns solver stopping rules from an arbitrary `tol` into
    /// "`|p(x)|` is below its own evaluation noise".
    pub fn eval_with_error(&self, x: T) -> (T, T) {
        if N == 0 {
            return (T::ZERO, T::ZERO);
        }
        let two = T::from_u32(2);
        let u = T::EPSILON / two; // unit roundoff
        let ax = x.abs();
        let mut y = self.c[N - 1];
        let mut mu = y.abs() / two;
        for &k in self.c.iter().rev().skip(1) {
            y = y.mul_add_fast(x, k);
            mu = mu * ax + y.abs();
        }
        (y, u * (two * mu - y.abs()))
    }
}

/// Packs solver output (finite roots plus `NAN`/infinite sentinels for slots
/// without a real root) into a counted, ascending [`Roots`].
fn pack_roots<T: Real, const MAX: usize>(candidates: [T; MAX]) -> Roots<T, MAX> {
    let mut buf = [T::ZERO; MAX];
    let mut len = 0;
    for r in candidates {
        if Real::is_finite(r) {
            buf[len] = r;
            len += 1;
        }
    }
    // Insertion sort of the live prefix, ascending. All entries are finite,
    // so `>` is a total order here.
    let live = &mut buf[..len];
    for i in 1..len {
        let mut j = i;
        while j > 0 && live[j - 1] > live[j] {
            live.swap(j - 1, j);
            j -= 1;
        }
    }
    Roots::from_buf(buf, len)
}

impl<T: Real> Polynomial<T, 2> {
    /// Real roots of the linear polynomial `c[1]·x + c[0]`, ascending;
    /// `len()` is the number of real roots found (0 if the leading
    /// coefficient is zero).
    pub fn roots(&self, _tol: T) -> Roots<T, 1> {
        pack_roots([-self.c[0] / self.c[1]])
    }
}

impl<T: Real> Polynomial<T, 3> {
    /// Real roots of the quadratic polynomial, ascending; `len()` is the
    /// number of real roots found.
    pub fn roots(&self, _tol: T) -> Roots<T, 2> {
        pack_roots(solvers::blinn::roots_quadratic(self))
    }
}

impl<T: Real> Polynomial<T, 4> {
    /// Real roots of the cubic polynomial, ascending; `len()` is the number
    /// of real roots found.
    pub fn roots(&self, tol: T) -> Roots<T, 3> {
        pack_roots(solvers::yuksel::roots_cubic(self, tol))
    }
}

impl<T: Real> Polynomial<T, 5> {
    /// Real roots of the quartic polynomial, ascending; `len()` is the number
    /// of real roots found.
    pub fn roots(&self, tol: T) -> Roots<T, 4> {
        pack_roots(solvers::yuksel::roots_quartic(self, tol))
    }
}

// Writes straight to the `Formatter` (no allocation) so it works in `no_std`.
// Prints highest-degree-first even though storage is ascending: presentation
// is independent of the storage convention.
impl<T: core::fmt::Display, const N: usize> core::fmt::Display for Polynomial<T, N> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        for (i, c) in self.c.iter().enumerate().rev() {
            if i > 0 {
                write!(f, "{}×x^{} + ", c, i)?;
            } else {
                write!(f, "{}", c)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::complex::{c32, c64};

    use super::*;

    #[test]
    fn test_c32_polynomials() {
        // p(x) = x² + 2x + 3 in ascending storage: c[i] multiplies x^i.
        let a: c32 = c32::new(1.0, 0.0);
        let b: c32 = c32::new(2.0, 0.0);
        let c: c32 = c32::new(3.0, 0.0);
        let p: Polynomial<c32, 3> = [c, b, a].into();

        // Horner evaluation works for complex coefficients through `Scalar`.
        assert_eq!(p.eval(c32::new(0.0, 0.0)), c);
        assert_eq!(p.eval(c32::new(1.0, 0.0)), c32::new(6.0, 0.0));
        assert_eq!(p.eval(c32::new(0.0, 1.0)), c32::new(2.0, 2.0));
    }

    #[test]
    fn test_f64_polynomials() {
        let p_0 = Polynomial::new([1.0]);
        let p_1 = Polynomial::new([2.0, 1.0]); // x + 2
        let p_2 = Polynomial::new([3.0, 2.0, 1.0]); // x² + 2x + 3
        let p_3: Polynomial<f64, 4> = [4.0, 3.0, 2.0, 1.0].into(); // x³ + 2x² + 3x + 4

        assert_eq!(p_0.c, [1.0]);
        assert_eq!(p_1.c, [2.0, 1.0]);
        assert_eq!(p_2.c, [3.0, 2.0, 1.0]);
        assert_eq!(p_3.c, [4.0, 3.0, 2.0, 1.0]);

        // Display prints highest-degree-first regardless of storage order.
        assert_eq!(p_0.to_string(), "1");
        assert_eq!(p_1.to_string(), "1×x^1 + 2");
        assert_eq!(p_2.to_string(), "1×x^2 + 2×x^1 + 3");
        assert_eq!(p_3.to_string(), "1×x^3 + 2×x^2 + 3×x^1 + 4");

        assert_eq!(p_0.eval(-3.0), 1.0);
        assert_eq!(p_0.eval(0.0), 1.0);
        assert_eq!(p_0.eval(1.0), 1.0);
        assert_eq!(p_0.eval(2.0), 1.0);

        assert_eq!(p_1.eval(-3.0), -1.0);
        assert_eq!(p_1.eval(0.0), 2.0);
        assert_eq!(p_1.eval(1.0), 3.0);
        assert_eq!(p_1.eval(2.0), 4.0);

        assert_eq!(p_2.eval(-3.0), 6.0);
        assert_eq!(p_2.eval(0.0), 3.0);
        assert_eq!(p_2.eval(1.0), 6.0);
        assert_eq!(p_2.eval(2.0), 11.0);

        assert_eq!(p_3.eval(-3.0), -14.0);
        assert_eq!(p_3.eval(0.0), 4.0);
        assert_eq!(p_3.eval(1.0), 10.0);
        assert_eq!(p_3.eval(2.0), 26.0);
    }

    #[test]
    fn ascending_convention_index_is_power() {
        // c[i] multiplies x^i: an asymmetric polynomial catches order bugs.
        // p(x) = 7 + 5x³.
        let p = Polynomial::new([7.0, 0.0, 0.0, 5.0]);
        assert_eq!(p.eval(0.0), 7.0);
        assert_eq!(p.eval(1.0), 12.0);
        assert_eq!(p.eval(2.0), 47.0);
        assert_eq!(p.to_string(), "5×x^3 + 0×x^2 + 0×x^1 + 7");
    }

    #[test]
    fn eval_degenerate_empty() {
        let p = Polynomial::<f64, 0>::new([]);
        assert_eq!(p.eval(3.0), 0.0);
        assert_eq!(p.eval_at(3.0), 0.0);
        assert_eq!(p.eval_with_error(3.0), (0.0, 0.0));
    }

    #[test]
    fn eval_higher_degree() {
        // Horner handles arbitrary N; 3x^5 + 1 at x = 2 is 97.
        let p = Polynomial::new([1.0, 0.0, 0.0, 0.0, 0.0, 3.0]);
        assert_eq!(p.eval(2.0), 97.0);
    }

    #[test]
    fn eval_at_matches_eval_for_scalar_points() {
        let p = Polynomial::new([4.0, -3.0, 2.0, -1.0]);
        for x in [-2.5, -1.0, 0.0, 0.5, 3.0] {
            assert_eq!(p.eval_at(x), p.eval(x));
        }

        // Integer coefficients over a plain Ring (no Scalar impl involved).
        let q = Polynomial::new([1i64, 2, 3]);
        assert_eq!(q.eval_at(10i64), 321);
    }

    #[test]
    fn eval_at_complex_point_of_real_polynomial() {
        // The whole point of Algebra<T>: real coefficients, complex point.
        // p(x) = x² + 1 vanishes at x = i.
        let p = Polynomial::new([1.0, 0.0, 1.0]);
        let z: c64 = p.eval_at(c64::new(0.0, 1.0));
        assert_eq!(z, c64::new(0.0, 0.0));

        // p(x) = x² - 2x + 2 at x = 1 + i: (1+i)² - 2(1+i) + 2 = 2i - 2i = 0.
        let p = Polynomial::new([2.0, -2.0, 1.0]);
        assert_eq!(p.eval_at(c64::new(1.0, 1.0)), c64::new(0.0, 0.0));

        // And a nonzero value, checked by hand: p(x) = x³ at x = i is -i.
        let p = Polynomial::new([0.0, 0.0, 0.0, 1.0]);
        assert_eq!(p.eval_at(c64::new(0.0, 1.0)), c64::new(0.0, -1.0));

        // Complex coefficients at a complex point (X = T through the blanket
        // Algebra instance).
        let q = Polynomial::new([c64::new(1.0, 1.0), c64::new(0.0, 2.0)]);
        assert_eq!(q.eval_at(c64::new(3.0, 0.0)), c64::new(1.0, 7.0));
    }

    #[test]
    fn eval_with_error_bound_dominates_true_error() {
        // Expanded (x-1)^6 is catastrophically ill-conditioned near x = 1:
        // f32 evaluation loses everything to cancellation. The running error
        // bound must dominate the true error (f64 Horner as reference).
        let p32 = Polynomial::<f32, 7>::new([1., -6., 15., -20., 15., -6., 1.]);
        let p64 = Polynomial::<f64, 7>::new([1., -6., 15., -20., 15., -6., 1.]);
        let mut checked = 0;
        for i in 0..=100 {
            let x64 = 0.99 + 0.0002 * (i as f64);
            let x32 = x64 as f32;
            let (y32, bound) = p32.eval_with_error(x32);
            let reference = p64.eval(x32 as f64);
            let true_err = (y32 as f64 - reference).abs();
            assert!(
                true_err <= bound as f64,
                "x = {}: true error {} exceeds bound {}",
                x32,
                true_err,
                bound
            );
            checked += 1;
        }
        assert_eq!(checked, 101);

        // The bound is small relative to a coarse |p|(|x|) magnitude yet
        // nonzero — i.e. it is actually a rounding-noise scale, not garbage.
        let (_, bound) = p32.eval_with_error(1.01);
        assert!(bound > 0.0 && bound < 1e-4);
    }

    #[test]
    fn eval_with_error_exact_evaluation() {
        // Small-integer coefficients at small-integer points are exact; the
        // value must match eval() and the bound must cover the (zero) error.
        let p = Polynomial::new([4.0, -3.0, 2.0, -1.0]);
        let (y, bound) = p.eval_with_error(2.0);
        assert_eq!(y, p.eval(2.0));
        assert!(bound >= 0.0);
        assert!(bound < 1e-13); // (2·deg+1)·u·μ scale for tiny μ
    }

    #[test]
    fn roots_1_linear() {
        let tol = 1e-7;

        // Linear polynomial p(x) = x + 1 (ascending: [1, 1]).
        let x = Polynomial::new([1., 1.]);
        let r = x.roots(tol);
        assert_eq!(r.len(), 1);
        assert_eq!(r.as_slice(), &[-1.]);

        // Degenerate leading coefficient (c[1] = 0): no isolated real root.
        let y = Polynomial::new([1., 0.]);
        let s = y.roots(tol);
        assert_eq!(s.len(), 0);
        assert!(s.is_empty());
    }

    #[test]
    fn roots_2_default() {
        let tol = 1e-7;

        // p(x) = 0x^2 + 1x + 1 with the single real root -1
        let x = Polynomial::new([1., 1., 0.]);
        let r = x.roots(tol);
        assert_eq!(r.len(), 1);
        assert_eq!(r.as_slice(), &[-1.]);

        // Quadratic p(x) = x^2 - x - 12 with roots -3, 4 (ascending)
        let x = Polynomial::new([-12., -1., 1.]);
        let r = x.roots(tol);
        assert_eq!(r.len(), 2);
        assert_eq!(r.as_slice(), &[-3., 4.]);

        // Quadratic p(x) = x^2 - 6x + 9 with root x = 3 of multiplicity 2
        let x = Polynomial::new([9., -6., 1.]);
        let r = x.roots(tol);
        assert_eq!(r.len(), 2);
        assert_eq!(r.as_slice(), &[3., 3.]);

        // Quadratic p(x) = x^2 - 3x + 5 with complex roots: no real roots.
        let x = Polynomial::new([5., -3., 1.]);
        let r = x.roots(tol);
        assert_eq!(r.len(), 0);
        assert!(r.is_empty());
    }

    #[test]
    fn roots_2_yuksel() {
        // p(x) = 0x^2 + 1x + 1 with root -1: the degenerate slot is -inf in
        // the raw solver output.
        let x = Polynomial::new([1., 1., 0.]);
        let r: [f64; 2] = solvers::yuksel::roots_quadratic(&x);
        assert_eq!(r[1], -1.);
        assert!(!r[0].is_finite());
        assert_eq!(r.len(), 2);

        // Quadratic p(x) = x^2 - x - 12 with roots 4,-3
        let x = Polynomial::new([-12., -1., 1.]);
        let r = solvers::yuksel::roots_quadratic(&x);
        assert_eq!(r[1], 4.);
        assert_eq!(r[0], -3.);

        // Quadratic p(x) = x^2 - 6x + 9 with root x = 3 with multiplicity 2
        let x = Polynomial::new([9., -6., 1.]);
        let r: [f64; 2] = solvers::yuksel::roots_quadratic(&x);
        assert_eq!(r[0], 3.);
        assert!(r[1].is_nan());

        // Quadratic p(x) = x^2 - 3x + 5 with complex roots
        let x = Polynomial::new([5., -3., 1.]);
        let r: [f64; 2] = solvers::yuksel::roots_quadratic(&x);
        assert!(r[0].is_nan());
        assert!(r[1].is_nan());
    }

    #[test]
    fn roots_3_generic() {
        let tol = f64::EPSILON;

        // Cubic p(x) = x^3 + 5x^2 - 14x + 0 with roots -7, 0, 2 (ascending).
        let x = Polynomial::new([0., -14., 5., 1.]);
        let r = x.roots(tol);
        assert_eq!(r.len(), 3);
        assert!((r[0] + 7.0).abs() < 5.0 * tol);
        assert!((r[1] - 0.0).abs() < 5.0 * tol);
        assert!((r[2] - 2.0).abs() < 5.0 * tol);

        // Cubic with a single real root: p(x) = x^3 + x - 2 = (x-1)(x^2+x+2).
        let x = Polynomial::new([-2., 1., 0., 1.]);
        let r = x.roots(tol);
        assert_eq!(r.len(), 1);
        assert!((r[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn roots_3_generic_f32() {
        // The solvers are generic over `Real` now; exercise f32.
        let tol = f32::EPSILON;
        let x = Polynomial::new([0_f32, -14., 5., 1.]);
        let r = x.roots(tol);
        assert_eq!(r.len(), 3);
        assert!((r[0] + 7.0).abs() < 10.0 * tol);
        assert!((r[1] - 0.0).abs() < 10.0 * tol);
        assert!((r[2] - 2.0).abs() < 10.0 * tol);
    }

    #[test]
    fn roots_3_blinn() {
        let tol = f64::EPSILON;

        // Cubic p(x) = x^3 + 5x^2 - 14x + 0 with roots -7, 0, 2; Blinn's
        // solver returns them in its native (descending) order.
        let x = Polynomial::new([0., -14., 5., 1.]);
        let r = solvers::blinn::roots_cubic(&x);
        assert!((r[0] - 2.0).abs() < 5.0 * tol);
        assert!((r[1] - 0.0).abs() < 5.0 * tol);
        assert!((r[2] + 7.0).abs() < 5.0 * tol);
        assert_eq!(r.len(), 3);
    }

    #[test]
    fn roots_3_yuksel() {
        let tol = f64::EPSILON;

        // Cubic p(x) = x^3 + 5x^2 - 14x + 0 with roots -7, 0, 2
        let x = Polynomial::new([0., -14., 5., 1.]);
        let r = solvers::yuksel::roots_cubic(&x, tol);
        assert_eq!(r[0], -7.0);
        assert_eq!(r[1], 0.);
        assert_eq!(r[2], 2.0);
        assert_eq!(r.len(), 3);
    }

    #[test]
    fn roots_4_four_real() {
        let tol = f64::EPSILON;

        // p(x) = (x^2 - 1)(x^2 - 4) = x^4 - 5x^2 + 4, roots -2, -1, 1, 2.
        let x = Polynomial::new([4., 0., -5., 0., 1.]);
        let r = x.roots(tol);
        assert_eq!(r.len(), 4);
        let expected = [-2., -1., 1., 2.];
        for (found, want) in r.iter().zip(expected) {
            assert!(
                (found - want).abs() < 1e-9,
                "root {} != expected {}",
                found,
                want
            );
        }
    }

    #[test]
    fn roots_4_two_real() {
        let tol = f64::EPSILON;

        // p(x) = (x^2 - 1)(x^2 + 1) = x^4 - 1, real roots -1, 1.
        let x = Polynomial::new([-1., 0., 0., 0., 1.]);
        let r = x.roots(tol);
        assert_eq!(r.len(), 2);
        assert!((r[0] + 1.0).abs() < 1e-9);
        assert!((r[1] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn roots_4_no_real() {
        let tol = f64::EPSILON;

        // p(x) = x^4 + 1 has no real roots.
        let x = Polynomial::new([1., 0., 0., 0., 1.]);
        let r = x.roots(tol);
        assert_eq!(r.len(), 0);
        assert!(r.is_empty());
    }

    #[test]
    fn roots_equality_ignores_dead_slots() {
        let tol = 1e-7;
        // Same live roots, different degenerate storage histories.
        let a = Polynomial::new([-12., -1., 1.]).roots(tol);
        let b = Polynomial::new([-24., -2., 2.]).roots(tol);
        assert_eq!(a, b);
    }
}
