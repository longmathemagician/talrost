//! Dense univariate polynomials with compile-time degree.
//!
//! `Polynomial<T, N>` stores `N` coefficients in descending-power order:
//! `c[0]·x^(N-1) + c[1]·x^(N-2) + … + c[N-1]`.
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

use crate::real::Real;
use crate::scalar::Scalar;
use crate::solvers;

pub use crate::roots::Roots;

#[derive(Copy, Clone, Debug)]
pub struct Polynomial<T, const N: usize> {
    pub c: [T; N],
}

impl<T, const N: usize> Polynomial<T, N> {
    pub const fn new(c: [T; N]) -> Self {
        Self { c }
    }
}

impl<T, const N: usize> From<[T; N]> for Polynomial<T, N> {
    fn from(c: [T; N]) -> Self {
        Self::new(c)
    }
}

impl<T: Scalar, const N: usize> Polynomial<T, N> {
    /// Evaluates the polynomial at `x` by Horner's method (monomorphization
    /// fully unrolls the fold for each `N`).
    ///
    /// A degenerate `Polynomial<T, 0>` has no coefficients and evaluates to
    /// zero (the empty sum).
    pub fn eval(&self, x: T) -> T {
        if N == 0 {
            return T::ZERO;
        }
        self.c
            .iter()
            .skip(1)
            .fold(self.c[0], |acc, &k| acc.mul_add(x, k))
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
    /// Real roots of the linear polynomial `c[0]·x + c[1]`, ascending;
    /// `len()` is the number of real roots found (0 if the leading
    /// coefficient is zero).
    pub fn roots(&self, _tol: T) -> Roots<T, 1> {
        pack_roots([-self.c[1] / self.c[0]])
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

impl<T: core::fmt::Display, const N: usize> core::fmt::Display for Polynomial<T, N> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut output = String::new();
        for i in 0..N {
            if i != (N - 1) {
                output.push_str(&format!("{}×x^{} + ", self.c[i], (N - 1) - i));
            } else {
                output.push_str(&format!("{}", self.c[i]));
            }
        }
        f.write_str(&output)
    }
}

#[cfg(test)]
mod tests {
    use crate::complex::c32;

    use super::*;

    #[test]
    fn test_c32_polynomials() {
        let a: c32 = c32::new(1.0, 0.0);
        let b: c32 = c32::new(2.0, 0.0);
        let c: c32 = c32::new(3.0, 0.0);
        let p: Polynomial<c32, 3> = [a, b, c].into();

        // Horner evaluation works for complex coefficients through `Scalar`.
        assert_eq!(p.eval(c32::new(0.0, 0.0)), c);
        assert_eq!(p.eval(c32::new(1.0, 0.0)), c32::new(6.0, 0.0));
        assert_eq!(p.eval(c32::new(0.0, 1.0)), c32::new(2.0, 2.0));
    }

    #[test]
    fn test_f64_polynomials() {
        let p_0 = Polynomial::new([1.0]);
        let p_1 = Polynomial::new([1.0, 2.0]);
        let p_2 = Polynomial::new([1.0, 2.0, 3.0]);
        let p_3: Polynomial<f64, 4> = [1.0, 2.0, 3.0, 4.0].into();

        assert_eq!(p_0.c, [1.0]);
        assert_eq!(p_1.c, [1.0, 2.0]);
        assert_eq!(p_2.c, [1.0, 2.0, 3.0]);
        assert_eq!(p_3.c, [1.0, 2.0, 3.0, 4.0]);

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
    fn eval_degenerate_empty() {
        let p = Polynomial::<f64, 0>::new([]);
        assert_eq!(p.eval(3.0), 0.0);
    }

    #[test]
    fn eval_higher_degree() {
        // Horner handles arbitrary N; x^5 + 1 at x = 2 is 33.
        let p = Polynomial::new([1.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
        assert_eq!(p.eval(2.0), 33.0);
    }

    #[test]
    fn roots_1_linear() {
        let tol = 1e-7;

        // Linear polynomial p(x) = x + 1
        let x = Polynomial::new([1., 1.]);
        let r = x.roots(tol);
        assert_eq!(r.len(), 1);
        assert_eq!(r.as_slice(), &[-1.]);

        // Degenerate leading coefficient: no isolated real root found.
        let y = Polynomial::new([0., 1.]);
        let s = y.roots(tol);
        assert_eq!(s.len(), 0);
        assert!(s.is_empty());
    }

    #[test]
    fn roots_2_default() {
        let tol = 1e-7;

        // p(x) = 0x^2 + 1x + 1 with the single real root -1
        let x = Polynomial::new([0., 1., 1.]);
        let r = x.roots(tol);
        assert_eq!(r.len(), 1);
        assert_eq!(r.as_slice(), &[-1.]);

        // Quadratic p(x) = x^2 - x - 12 with roots -3, 4 (ascending)
        let x = Polynomial::new([1., -1., -12.]);
        let r = x.roots(tol);
        assert_eq!(r.len(), 2);
        assert_eq!(r.as_slice(), &[-3., 4.]);

        // Quadratic p(x) = x^2 - 6x + 9 with root x = 3 of multiplicity 2
        let x = Polynomial::new([1., -6., 9.]);
        let r = x.roots(tol);
        assert_eq!(r.len(), 2);
        assert_eq!(r.as_slice(), &[3., 3.]);

        // Quadratic p(x) = x^2 - 3x + 5 with complex roots: no real roots.
        let x = Polynomial::new([1., -3., 5.]);
        let r = x.roots(tol);
        assert_eq!(r.len(), 0);
        assert!(r.is_empty());
    }

    #[test]
    fn roots_2_yuksel() {
        // p(x) = 0x^2 + 1x + 1 with root -1: the degenerate slot is -inf in
        // the raw solver output.
        let x = Polynomial::new([0., 1., 1.]);
        let r: [f64; 2] = solvers::yuksel::roots_quadratic(&x);
        assert_eq!(r[1], -1.);
        assert_eq!(r[0].is_finite(), false);
        assert_eq!(r.len(), 2);

        // Quadratic p(x) = x^2 - x - 12 with roots 4,-3
        let x = Polynomial::new([1., -1., -12.]);
        let r = solvers::yuksel::roots_quadratic(&x);
        assert_eq!(r[1], 4.);
        assert_eq!(r[0], -3.);

        // Quadratic p(x) = x^2 - 6x + 9 with root x = 3 with multiplicity 2
        let x = Polynomial::new([1., -6., 9.]);
        let r: [f64; 2] = solvers::yuksel::roots_quadratic(&x);
        assert_eq!(r[0], 3.);
        assert_eq!(r[1].is_nan(), true);

        // Quadratic p(x) = x^2 - 3x + 5 with complex roots
        let x = Polynomial::new([1., -3., 5.]);
        let r: [f64; 2] = solvers::yuksel::roots_quadratic(&x);
        assert_eq!(r[0].is_nan(), true);
        assert_eq!(r[1].is_nan(), true);
    }

    #[test]
    fn roots_3_generic() {
        let tol = f64::EPSILON;

        // Cubic p(x) = 1x^3 + 5x^2 + -14x + 0 with roots -7, 0, 2 (ascending).
        let x = Polynomial::new([1., 5., -14., 0.]);
        let r = x.roots(tol);
        assert_eq!(r.len(), 3);
        assert!((r[0] + 7.0).abs() < 5.0 * tol);
        assert!((r[1] - 0.0).abs() < 5.0 * tol);
        assert!((r[2] - 2.0).abs() < 5.0 * tol);

        // Cubic with a single real root: p(x) = x^3 + x - 2 = (x-1)(x^2+x+2).
        let x = Polynomial::new([1., 0., 1., -2.]);
        let r = x.roots(tol);
        assert_eq!(r.len(), 1);
        assert!((r[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn roots_3_generic_f32() {
        // The solvers are generic over `Real` now; exercise f32.
        let tol = f32::EPSILON;
        let x = Polynomial::new([1_f32, 5., -14., 0.]);
        let r = x.roots(tol);
        assert_eq!(r.len(), 3);
        assert!((r[0] + 7.0).abs() < 10.0 * tol);
        assert!((r[1] - 0.0).abs() < 10.0 * tol);
        assert!((r[2] - 2.0).abs() < 10.0 * tol);
    }

    #[test]
    fn roots_3_blinn() {
        let tol = f64::EPSILON;

        // Cubic p(x) = 1x^3 + 5x^2 + -14x + 0 with roots -7, 0, 2; Blinn's
        // solver returns them in its native (descending) order.
        let x = Polynomial::new([1., 5., -14., 0.]);
        let r = solvers::blinn::roots_cubic(&x);
        assert_eq!((r[0] - 2.0).abs() < 5.0 * tol, true);
        assert_eq!((r[1] - 0.0).abs() < 5.0 * tol, true);
        assert_eq!((r[2] + 7.0).abs() < 5.0 * tol, true);
        assert_eq!(r.len(), 3);
    }

    #[test]
    fn roots_3_yuksel() {
        let tol = f64::EPSILON;

        // Cubic p(x) = 1x^3 + 5x^2 + -14x + 0 with roots -7, 0, 2
        let x = Polynomial::new([1., 5., -14., 0.]);
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
        let x = Polynomial::new([1., 0., -5., 0., 4.]);
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
        let x = Polynomial::new([1., 0., 0., 0., -1.]);
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
        let a = Polynomial::new([1., -1., -12.]).roots(tol);
        let b = Polynomial::new([2., -2., -24.]).roots(tol);
        assert_eq!(a, b);
    }
}
