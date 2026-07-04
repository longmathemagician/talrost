//! Multivariate sparse polynomials with compile-time shape.
//!
//! A sparse polynomial in `NV` variables is a list of `TERMS` terms, each a
//! coefficient and a [`Monomial`] (exponent vector). Everything is
//! stack-only and monomorphized: `TERMS` is a compile-time capacity, and a
//! system pads every row to a common `MAXT` with zero-coefficient terms —
//! the deliberate answer to heterogeneous term counts vs const generics
//! (small waste, uniform type, no allocation).
//!
//! Exponents are `i32`, not `u32`: after the torus transforms of a
//! polyhedral homotopy the monomials become Laurent (negative exponents).
//! Plain [`MPoly::eval_at`] is exponentiation over a [`Ring`] and
//! `debug_assert`s non-negative exponents; a `Field`-bounded Laurent eval
//! can come later.
//!
//! Like the univariate [`crate::polynomial::Polynomial`] (§5.1), exponent
//! semantics are positional: `exps[i]` is the power of variable `i`.

use crate::algebra::{Algebra, Ring};
use crate::dual::DualN;
use crate::matrix::Matrix;

/// An exponent vector: `x0^exps[0] · x1^exps[1] · …`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Monomial<const NV: usize> {
    /// The exponent of variable `i` at position `i`. `i32`, not `u32`:
    /// Laurent (negative) exponents arise after torus transforms.
    pub exps: [i32; NV],
}

impl<const NV: usize> Monomial<NV> {
    /// Builds a monomial from its exponent vector.
    pub const fn new(exps: [i32; NV]) -> Self {
        Self { exps }
    }
}

/// A sparse multivariate polynomial: `TERMS` terms of coefficient ×
/// monomial. Zero-coefficient terms are legal and simply contribute nothing
/// — they are the padding mechanism for [`MSystem`].
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct MPoly<T, const NV: usize, const TERMS: usize> {
    /// The coefficient of each term (zero coefficients mark padding).
    pub coeffs: [T; TERMS],
    /// The monomial (exponent vector) of each term, paired with `coeffs`
    /// positionally.
    pub support: [Monomial<NV>; TERMS],
}

/// A square-ish system of `NEQ` sparse polynomials in `NV` variables, every
/// row padded to `MAXT` terms with zero coefficients.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct MSystem<T, const NV: usize, const NEQ: usize, const MAXT: usize> {
    /// The equations, one padded [`MPoly`] per row.
    pub polys: [MPoly<T, NV, MAXT>; NEQ],
}

/// `x^e` by exponentiation-by-squaring (`O(log e)` multiplies).
fn pow_u32<X: Ring>(x: X, mut e: u32) -> X {
    let mut acc = X::ONE;
    let mut base = x;
    while e > 0 {
        if e & 1 == 1 {
            acc *= base;
        }
        base *= base;
        e >>= 1;
    }
    acc
}

/// `k·t` over the additive group of any [`Ring`] (double-and-add), so
/// structural derivatives don't need an integer-to-`T` conversion.
fn int_scale<T: Ring>(t: T, k: i32) -> T {
    let neg = k < 0;
    let mut k = k.unsigned_abs();
    let mut base = t;
    let mut acc = T::ZERO;
    while k > 0 {
        if k & 1 == 1 {
            acc += base;
        }
        base = base + base;
        k >>= 1;
    }
    if neg {
        -acc
    } else {
        acc
    }
}

impl<T: Ring, const NV: usize, const TERMS: usize> MPoly<T, NV, TERMS> {
    /// Builds a sparse polynomial from positionally-paired coefficients and
    /// monomials. Pad unused slots with zero coefficients (any exponents).
    pub const fn new(coeffs: [T; TERMS], support: [Monomial<NV>; TERMS]) -> Self {
        Self { coeffs, support }
    }

    /// Evaluates at a point of any [`Algebra`] over the coefficient ring:
    /// per term, `coeff · ∏ xᵢ^eᵢ` with exponentiation-by-squaring. The same
    /// body serves plain values (`X = T`), complex points of real systems,
    /// and [`DualN`] points (derivatives).
    ///
    /// Negative (Laurent) exponents are rejected by `debug_assert`: they
    /// only arise after torus transforms, which live in the solver layer.
    pub fn eval_at<X: Algebra<T>>(&self, x: &[X; NV]) -> X {
        let mut sum = X::ZERO;
        for (coeff, mono) in self.coeffs.iter().zip(self.support.iter()) {
            let mut term = X::ONE;
            for (xi, &e) in x.iter().zip(mono.exps.iter()) {
                debug_assert!(
                    e >= 0,
                    "negative exponent in plain eval (Laurent monomials need a torus transform)"
                );
                if e != 0 {
                    term *= pow_u32(*xi, e as u32);
                }
            }
            sum += term * *coeff;
        }
        sum
    }

    /// The structural partial derivative `∂/∂x_j`: per term, coefficient
    /// `coeff·eⱼ` and exponent `eⱼ − 1`. The term count is preserved under
    /// padding, so the type doesn't change; terms with `eⱼ = 0` get a zero
    /// coefficient (padding working as intended) and keep their exponents
    /// valid.
    ///
    /// This is the Jacobian fast path — no dual arithmetic, exact integer
    /// scaling — and it works for Laurent exponents too
    /// (`d/dx x⁻² = −2x⁻³`).
    pub fn partial(&self, j: usize) -> Self {
        let mut coeffs = self.coeffs;
        let mut support = self.support;
        for (c, m) in coeffs.iter_mut().zip(support.iter_mut()) {
            let e = m.exps[j];
            if e == 0 {
                *c = T::ZERO;
            } else {
                *c = int_scale(*c, e);
                m.exps[j] = e - 1;
            }
        }
        Self { coeffs, support }
    }

    /// Value and full gradient in one sweep via [`DualN<T, NV>`]: seed each
    /// coordinate as its own variable and read the derivative slots back.
    pub fn eval_grad(&self, x: &[T; NV]) -> (T, [T; NV]) {
        let mut xs = [DualN::<T, NV>::constant(T::ZERO); NV];
        for (k, (xk, &xv)) in xs.iter_mut().zip(x.iter()).enumerate() {
            *xk = DualN::variable(xv, k);
        }
        let r = self.eval_at(&xs);
        (r.val, r.der)
    }
}

impl<T: Ring, const NV: usize, const NEQ: usize, const MAXT: usize> MSystem<T, NV, NEQ, MAXT> {
    /// Builds a system from its equations (each already padded to `MAXT`
    /// terms).
    pub const fn new(polys: [MPoly<T, NV, MAXT>; NEQ]) -> Self {
        Self { polys }
    }

    /// Evaluates every equation at the same point.
    pub fn eval<X: Algebra<T>>(&self, x: &[X; NV]) -> [X; NEQ] {
        let mut out = [X::ZERO; NEQ];
        for (o, p) in out.iter_mut().zip(self.polys.iter()) {
            *o = p.eval_at(x);
        }
        out
    }

    /// Values and the `NEQ×NV` Jacobian in one sweep: each equation is
    /// evaluated once in `DualN<T, NV>` arithmetic and its derivative slots
    /// become a Jacobian row. This is the Newton corrector's per-iteration
    /// workload (pair it with [`Matrix::lu`] + `Lu::solve`).
    pub fn eval_jacobian(&self, x: &[T; NV]) -> ([T; NEQ], Matrix<T, NEQ, NV>) {
        let mut xs = [DualN::<T, NV>::constant(T::ZERO); NV];
        for (k, (xk, &xv)) in xs.iter_mut().zip(x.iter()).enumerate() {
            *xk = DualN::variable(xv, k);
        }
        let mut vals = [T::ZERO; NEQ];
        let mut jac = Matrix::<T, NEQ, NV>::ZERO;
        for ((v, row), p) in vals.iter_mut().zip(jac.e.iter_mut()).zip(self.polys.iter()) {
            let r = p.eval_at(&xs);
            *v = r.val;
            *row = r.der;
        }
        (vals, jac)
    }
}

// Writes straight to the `Formatter` (no allocation) so it works in `no_std`.
// "x0^2·x1", or "1" for the constant monomial.
impl<const NV: usize> core::fmt::Display for Monomial<NV> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut wrote = false;
        for (i, &e) in self.exps.iter().enumerate() {
            if e == 0 {
                continue;
            }
            if wrote {
                f.write_str("·")?;
            }
            wrote = true;
            if e == 1 {
                write!(f, "x{}", i)?;
            } else {
                write!(f, "x{}^{}", i, e)?;
            }
        }
        if !wrote {
            f.write_str("1")?;
        }
        Ok(())
    }
}

// "3·x0^2·x1 + -1·x1^3"; zero-coefficient (padding) terms are skipped, in
// storage order, and the zero polynomial prints "0". Allocation-free.
impl<T, const NV: usize, const TERMS: usize> core::fmt::Display for MPoly<T, NV, TERMS>
where
    T: Ring + PartialEq + core::fmt::Display,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut wrote = false;
        for (c, m) in self.coeffs.iter().zip(self.support.iter()) {
            if *c == T::ZERO {
                continue; // padding
            }
            if wrote {
                f.write_str(" + ")?;
            }
            wrote = true;
            write!(f, "{}", c)?;
            if m.exps.iter().any(|&e| e != 0) {
                write!(f, "·{}", m)?;
            }
        }
        if !wrote {
            f.write_str("0")?;
        }
        Ok(())
    }
}

// One equation per line, "= 0" suffixed.
impl<T, const NV: usize, const NEQ: usize, const MAXT: usize> core::fmt::Display
    for MSystem<T, NV, NEQ, MAXT>
where
    T: Ring + PartialEq + core::fmt::Display,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        for p in self.polys.iter() {
            writeln!(f, "{} = 0", p)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::complex::c64;

    /// f(x, y) = 3x²y − 2y³ + 5 (three terms, two variables).
    fn sample() -> MPoly<f64, 2, 3> {
        MPoly::new(
            [3.0, -2.0, 5.0],
            [
                Monomial::new([2, 1]),
                Monomial::new([0, 3]),
                Monomial::new([0, 0]),
            ],
        )
    }

    #[test]
    fn eval_known_values() {
        let p = sample();
        // f(1, 1) = 3 - 2 + 5 = 6; f(2, -1) = -12 + 2 + 5 = -5; f(0, 2) = -16 + 5.
        assert_eq!(p.eval_at(&[1.0, 1.0]), 6.0);
        assert_eq!(p.eval_at(&[2.0, -1.0]), -5.0);
        assert_eq!(p.eval_at(&[0.0, 2.0]), -11.0);

        // Integer coefficients over a plain Ring.
        let q = MPoly::<i64, 2, 2>::new([2, -7], [Monomial::new([3, 0]), Monomial::new([1, 1])]);
        assert_eq!(q.eval_at(&[2i64, 3]), 16 - 42);
    }

    #[test]
    fn eval_exponentiation_by_squaring() {
        // x^10 with a single term: 2^10 = 1024.
        let p = MPoly::<f64, 1, 1>::new([1.0], [Monomial::new([10])]);
        assert_eq!(p.eval_at(&[2.0]), 1024.0);
        // Exact for high odd powers too.
        let p = MPoly::<f64, 1, 1>::new([1.0], [Monomial::new([13])]);
        assert_eq!(p.eval_at(&[2.0]), 8192.0);
        // x^0 = 1 even at x = 0 (the empty product).
        let p = MPoly::<f64, 1, 1>::new([7.0], [Monomial::new([0])]);
        assert_eq!(p.eval_at(&[0.0]), 7.0);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "negative exponent")]
    fn eval_rejects_laurent_exponents_in_debug() {
        let p = MPoly::<f64, 1, 1>::new([1.0], [Monomial::new([-1])]);
        let _ = p.eval_at(&[2.0]);
    }

    #[test]
    fn eval_at_complex_point_of_real_mpoly() {
        // f(x, y) = x² + y² at (i, 1) = 0 — real coefficients, complex point,
        // through the Algebra<f64> instance of Complex.
        let p = MPoly::<f64, 2, 2>::new([1.0, 1.0], [Monomial::new([2, 0]), Monomial::new([0, 2])]);
        let i = c64::new(0.0, 1.0);
        let one = c64::new(1.0, 0.0);
        assert_eq!(p.eval_at(&[i, one]), c64::new(0.0, 0.0));

        // Complex coefficients at complex points (X = T).
        let q = MPoly::<c64, 1, 2>::new(
            [c64::new(0.0, 1.0), c64::new(1.0, 0.0)],
            [Monomial::new([1]), Monomial::new([0])],
        );
        assert_eq!(q.eval_at(&[i]), c64::new(0.0, 0.0) + i * i + one);
    }

    #[test]
    fn structural_partial() {
        let p = sample(); // 3x²y − 2y³ + 5
        let px = p.partial(0); // 6xy
        let py = p.partial(1); // 3x² − 6y²

        assert_eq!(px.eval_at(&[2.0, -1.0]), -12.0);
        assert_eq!(py.eval_at(&[2.0, -1.0]), 12.0 - 6.0);

        // Term count (the type) is preserved; dead terms carry zero coeffs.
        assert_eq!(px.coeffs, [6.0, 0.0, 0.0]);
        assert_eq!(px.support[0], Monomial::new([1, 1]));
        // Exponents of killed terms stay untouched (and valid).
        assert_eq!(px.support[1], Monomial::new([0, 3]));

        // Second partials commute: f_xy == f_yx.
        let pxy = px.partial(1);
        let pyx = py.partial(0);
        for pt in [[1.0, 1.0], [2.0, -1.0], [-0.5, 3.0]] {
            assert_eq!(pxy.eval_at(&pt), pyx.eval_at(&pt));
        }
    }

    #[test]
    fn partial_equals_dualn_gradient() {
        let p = sample();
        for pt in [[1.0, 1.0], [2.0, -1.0], [-0.5, 3.0], [0.0, 0.0]] {
            let (val, grad) = p.eval_grad(&pt);
            assert_eq!(val, p.eval_at(&pt));
            assert_eq!(grad[0], p.partial(0).eval_at(&pt));
            assert_eq!(grad[1], p.partial(1).eval_at(&pt));
        }
    }

    #[test]
    fn padded_rows_contribute_nothing() {
        // The same polynomial with and without padding evaluates identically.
        let dense =
            MPoly::<f64, 2, 2>::new([1.0, 2.0], [Monomial::new([1, 0]), Monomial::new([0, 1])]);
        let padded = MPoly::<f64, 2, 4>::new(
            [1.0, 2.0, 0.0, 0.0],
            [
                Monomial::new([1, 0]),
                Monomial::new([0, 1]),
                Monomial::new([0, 0]),
                Monomial::new([5, 7]), // garbage exponents under a zero coeff
            ],
        );
        for pt in [[1.0, 1.0], [3.0, -2.0], [0.5, 0.25]] {
            assert_eq!(dense.eval_at(&pt), padded.eval_at(&pt));
        }
        // Gradients too: padding must not leak into derivative slots.
        let (_, g_dense) = dense.eval_grad(&[3.0, -2.0]);
        let (_, g_padded) = padded.eval_grad(&[3.0, -2.0]);
        assert_eq!(g_dense, g_padded);
    }

    #[test]
    fn msystem_eval_and_jacobian() {
        // The 2×2 system f₁ = x² + y² − 5, f₂ = xy − 2
        // J = [[2x, 2y], [y, x]].
        let f1 = MPoly::<f64, 2, 3>::new(
            [1.0, 1.0, -5.0],
            [
                Monomial::new([2, 0]),
                Monomial::new([0, 2]),
                Monomial::new([0, 0]),
            ],
        );
        let f2 = MPoly::<f64, 2, 3>::new(
            [1.0, -2.0, 0.0],
            [
                Monomial::new([1, 1]),
                Monomial::new([0, 0]),
                Monomial::new([0, 0]), // padding
            ],
        );
        let sys = MSystem::new([f1, f2]);

        // (2, 1) is a root of the system.
        let vals = sys.eval(&[2.0, 1.0]);
        assert_eq!(vals, [0.0, 0.0]);

        let (vals, jac) = sys.eval_jacobian(&[2.0, 1.0]);
        assert_eq!(vals, [0.0, 0.0]);
        assert_eq!(jac, Matrix::new([[4.0, 2.0], [1.0, 2.0]]));

        // Away from the root, values and Jacobian both check out.
        let (vals, jac) = sys.eval_jacobian(&[3.0, -1.0]);
        assert_eq!(vals, [9.0 + 1.0 - 5.0, -3.0 - 2.0]);
        assert_eq!(jac, Matrix::new([[6.0, -2.0], [-1.0, 3.0]]));

        // One Newton step from a nearby point converges toward (2, 1):
        // Δx = J⁻¹·(−f), the tracker's corrector inner loop.
        let x0 = [2.1, 0.9];
        let (h, j) = sys.eval_jacobian(&x0);
        let delta = j
            .solve(&crate::vector::Vector::new([-h[0], -h[1]]))
            .unwrap();
        let x1 = [x0[0] + delta.b[0], x0[1] + delta.b[1]];
        let r0 = sys.eval(&x0);
        let r1 = sys.eval(&x1);
        let n0 = r0[0] * r0[0] + r0[1] * r0[1];
        let n1 = r1[0] * r1[0] + r1[1] * r1[1];
        assert!(
            n1 < n0 * 1e-2,
            "Newton step failed to contract: {} -> {}",
            n0,
            n1
        );
    }

    #[test]
    fn msystem_complex_jacobian() {
        // g(z, w) = (z² − w, zw − 1) at (i, 2i):
        // values: (−1 − 2i, 2i·i − 1 = −3), J = [[2z, −1], [w, z]].
        let g1 = MPoly::<c64, 2, 2>::new(
            [c64::new(1.0, 0.0), c64::new(-1.0, 0.0)],
            [Monomial::new([2, 0]), Monomial::new([0, 1])],
        );
        let g2 = MPoly::<c64, 2, 2>::new(
            [c64::new(1.0, 0.0), c64::new(-1.0, 0.0)],
            [Monomial::new([1, 1]), Monomial::new([0, 0])],
        );
        let sys = MSystem::new([g1, g2]);
        let z = c64::new(0.0, 1.0);
        let w = c64::new(0.0, 2.0);
        let (vals, jac) = sys.eval_jacobian(&[z, w]);
        assert_eq!(vals[0], z * z - w);
        assert_eq!(vals[1], z * w - c64::new(1.0, 0.0));
        assert_eq!(jac.e[0][0], z + z);
        assert_eq!(jac.e[0][1], c64::new(-1.0, 0.0));
        assert_eq!(jac.e[1][0], w);
        assert_eq!(jac.e[1][1], z);
    }

    #[test]
    fn display_no_alloc_format() {
        let p = sample();
        assert_eq!(p.to_string(), "3·x0^2·x1 + -2·x1^3 + 5");

        // Padding terms are skipped; the zero polynomial prints "0".
        let padded = MPoly::<f64, 2, 3>::new(
            [1.0, 0.0, 0.0],
            [
                Monomial::new([1, 1]),
                Monomial::new([0, 0]),
                Monomial::new([0, 0]),
            ],
        );
        assert_eq!(padded.to_string(), "1·x0·x1");
        assert_eq!(
            MPoly::<f64, 2, 2>::new([0.0; 2], [Monomial::new([0, 0]); 2]).to_string(),
            "0"
        );

        assert_eq!(Monomial::<3>::new([2, 0, 1]).to_string(), "x0^2·x2");
        assert_eq!(Monomial::<2>::new([0, 0]).to_string(), "1");

        let sys = MSystem::new([padded]);
        assert_eq!(sys.to_string(), "1·x0·x1 = 0\n");
    }
}
