//! Dual numbers: forward-mode automatic differentiation as ring elements.
//!
//! `Dual<T>` is `a + a′ε` with `ε² = 0`; arithmetic on the `val` part is
//! ordinary arithmetic while the `der` part transports the derivative through
//! every operation (the product rule is literally the `Mul` impl). Seeding
//! `der = 1` ([`Dual::variable`]) and running any generic computation yields
//! that computation's derivative with no symbolic work and no truncation
//! error.
//!
//! `DualN<T, K>` is the vector-mode version: `K` derivative slots
//! differentiate with respect to `K` independent variables in one sweep —
//! one evaluation of an `NEQ`-equation system in `DualN<T, NV>` produces the
//! full `NEQ×NV` Jacobian row by row.
//!
//! Both types implement the algebraic tower through [`Ring`] (and [`Field`]
//! when `T` is one), and [`Algebra<T>`] — so every generic routine in the
//! crate (`Polynomial::eval_at`, `MPoly::eval_at`, …) differentiates itself.
//!
//! # Division
//!
//! `Dual` numbers with a zero standard part have no multiplicative inverse
//! (`ε` is nilpotent), so `Field::recip` and `Div` require `val != 0` — the
//! same pragmatism as floats vis-à-vis the exact field axioms; dividing by a
//! zero-standard-part dual yields infinities/NaNs rather than panicking.

use core::iter::Sum;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::algebra::{Algebra, Field, Group, Monoid, Ring, Semiring};
use crate::complex::Complex;
use crate::element::Element;
use crate::real::Real;

/// A dual number `val + der·ε` with `ε² = 0`; forward-mode AD with a single
/// derivative slot.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Dual<T> {
    pub val: T,
    pub der: T,
}

/// A vector dual number `val + Σ der[k]·ε_k` with `ε_i ε_j = 0`;
/// forward-mode AD with `K` derivative slots (one per independent variable).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct DualN<T, const K: usize> {
    pub val: T,
    pub der: [T; K],
}

impl<T: Ring> Dual<T> {
    /// Seeds `x` as *the* variable: `der = 1`, so downstream `der` values are
    /// derivatives with respect to it.
    pub fn variable(x: T) -> Self {
        Self {
            val: x,
            der: T::ONE,
        }
    }

    /// Lifts `x` as a constant: `der = 0`.
    pub fn constant(x: T) -> Self {
        Self {
            val: x,
            der: T::ZERO,
        }
    }
}

impl<T: Ring, const K: usize> DualN<T, K> {
    /// Seeds `x` as the `k`-th of `K` variables: `der = e_k`.
    ///
    /// # Panics
    /// If `k >= K`.
    pub fn variable(x: T, k: usize) -> Self {
        let mut der = [T::ZERO; K];
        der[k] = T::ONE;
        Self { val: x, der }
    }

    /// Lifts `x` as a constant: `der = 0`.
    pub fn constant(x: T) -> Self {
        Self {
            val: x,
            der: [T::ZERO; K],
        }
    }
}

// ---------------------------------------------------------------------------
// Dual<T>: arithmetic
// ---------------------------------------------------------------------------

impl<T: Ring> Add for Dual<T> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            val: self.val + rhs.val,
            der: self.der + rhs.der,
        }
    }
}

impl<T: Ring> Sub for Dual<T> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            val: self.val - rhs.val,
            der: self.der - rhs.der,
        }
    }
}

// The product rule: (a + a′ε)(b + b′ε) = ab + (ab′ + a′b)ε.
impl<T: Ring> Mul for Dual<T> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self {
            val: self.val * rhs.val,
            der: self.val * rhs.der + self.der * rhs.val,
        }
    }
}

// The quotient rule, in the fused form (a′ − (a/b)·b′)/b. Requires a nonzero
// standard part in `rhs` (see the module docs).
impl<T: Field> Div for Dual<T> {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        let val = self.val / rhs.val;
        Self {
            val,
            der: (self.der - val * rhs.der) / rhs.val,
        }
    }
}

impl<T: Ring> Neg for Dual<T> {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            val: -self.val,
            der: -self.der,
        }
    }
}

impl<T: Ring> AddAssign for Dual<T> {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}
impl<T: Ring> SubAssign for Dual<T> {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}
impl<T: Ring> MulAssign for Dual<T> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}
impl<T: Field> DivAssign for Dual<T> {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl<T: Ring> Sum for Dual<T> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(
            Self {
                val: T::ZERO,
                der: T::ZERO,
            },
            |a, b| a + b,
        )
    }
}

// Mixed-scalar ops: `T` on the right-hand side acts as a constant (zero
// derivative). Coherent with the `Self`-RHS impls above because `T` can never
// unify with `Dual<T>` (occurs check).
impl<T: Ring> Add<T> for Dual<T> {
    type Output = Self;
    fn add(self, rhs: T) -> Self {
        Self {
            val: self.val + rhs,
            der: self.der,
        }
    }
}
impl<T: Ring> Sub<T> for Dual<T> {
    type Output = Self;
    fn sub(self, rhs: T) -> Self {
        Self {
            val: self.val - rhs,
            der: self.der,
        }
    }
}
impl<T: Ring> Mul<T> for Dual<T> {
    type Output = Self;
    fn mul(self, rhs: T) -> Self {
        Self {
            val: self.val * rhs,
            der: self.der * rhs,
        }
    }
}
impl<T: Field> Div<T> for Dual<T> {
    type Output = Self;
    fn div(self, rhs: T) -> Self {
        Self {
            val: self.val / rhs,
            der: self.der / rhs,
        }
    }
}

impl<T: Ring> From<T> for Dual<T> {
    fn from(value: T) -> Self {
        Self::constant(value)
    }
}

// ---------------------------------------------------------------------------
// Dual<T>: tower + Algebra instances
// ---------------------------------------------------------------------------

impl<T: Ring> Element for Dual<T> {}
impl<T: Ring> Monoid for Dual<T> {
    const ZERO: Self = Self {
        val: T::ZERO,
        der: T::ZERO,
    };
}
impl<T: Ring> Group for Dual<T> {}
impl<T: Ring> Semiring for Dual<T> {
    const ONE: Self = Self {
        val: T::ONE,
        der: T::ZERO,
    };
}
impl<T: Ring> Ring for Dual<T> {}

// Division requires a nonzero standard part (see the module docs); with that
// caveat the duals over a field form a (local) ring with well-defined recip.
impl<T: Field> Field for Dual<T> {
    fn recip(self) -> Self {
        // 1/(v + dε) = 1/v − (d/v²)ε.
        let r = self.val.recip();
        Self {
            val: r,
            der: -(self.der * r * r),
        }
    }
}

impl<T: Ring> Algebra<T> for Dual<T> {}

// Nested composition: a dual over the complex numbers is also an algebra over
// the *real* subfield, so real-coefficient polynomials can be evaluated at
// `Dual<Complex<F>>` points directly (∂/∂t of a complex path, the homotopy
// predictor's t-slot).
impl<F: Real> Add<F> for Dual<Complex<F>> {
    type Output = Self;
    fn add(self, rhs: F) -> Self {
        Self {
            val: self.val + rhs,
            der: self.der,
        }
    }
}
impl<F: Real> Mul<F> for Dual<Complex<F>> {
    type Output = Self;
    fn mul(self, rhs: F) -> Self {
        Self {
            val: self.val * rhs,
            der: self.der * rhs,
        }
    }
}
impl<F: Real> From<F> for Dual<Complex<F>> {
    fn from(value: F) -> Self {
        Self::constant(Complex::from(value))
    }
}
impl<F: Real> Algebra<F> for Dual<Complex<F>> {}

// ---------------------------------------------------------------------------
// Dual<F: Real>: chain-rule lifts
// ---------------------------------------------------------------------------

/// Chain-rule lifts of the [`Real`] transcendental methods, as inherent
/// methods (deliberately *not* a full `Real` impl: duals are not ordered and
/// have no meaningful `EPSILON`/`INFINITY`/rounding constants).
impl<F: Real> Dual<F> {
    /// `d(√x) = x′ / (2√x)`.
    pub fn sqrt(self) -> Self {
        let s = self.val.sqrt();
        Self {
            val: s,
            der: self.der / (s + s),
        }
    }

    /// `d(sin x) = x′·cos x`.
    pub fn sin(self) -> Self {
        let (s, c) = self.val.sin_cos();
        Self {
            val: s,
            der: self.der * c,
        }
    }

    /// `d(cos x) = −x′·sin x`.
    pub fn cos(self) -> Self {
        let (s, c) = self.val.sin_cos();
        Self {
            val: c,
            der: -(self.der * s),
        }
    }

    /// Both [`Dual::sin`] and [`Dual::cos`] from a single `sin_cos` call.
    pub fn sin_cos(self) -> (Self, Self) {
        let (s, c) = self.val.sin_cos();
        (
            Self {
                val: s,
                der: self.der * c,
            },
            Self {
                val: c,
                der: -(self.der * s),
            },
        )
    }

    /// `d(tan x) = x′·(1 + tan²x)`.
    pub fn tan(self) -> Self {
        let t = self.val.tan();
        Self {
            val: t,
            der: self.der * (F::ONE + t * t),
        }
    }

    /// `d(eˣ) = x′·eˣ`.
    pub fn exp(self) -> Self {
        let e = self.val.exp();
        Self {
            val: e,
            der: self.der * e,
        }
    }

    /// `d(ln x) = x′ / x`.
    pub fn ln(self) -> Self {
        Self {
            val: self.val.ln(),
            der: self.der / self.val,
        }
    }

    /// `d(xⁿ) = x′·n·xⁿ⁻¹`.
    pub fn powi(self, n: i32) -> Self {
        if n == 0 {
            return Self::constant(F::ONE);
        }
        let nf = if n < 0 {
            -F::from_u32(n.unsigned_abs())
        } else {
            F::from_u32(n as u32)
        };
        Self {
            val: self.val.powi(n),
            der: self.der * nf * self.val.powi(n - 1),
        }
    }

    /// `self·a + b` with the value part fused ([`Real::mul_add`]) and the
    /// derivative by the product rule.
    pub fn mul_add(self, a: Self, b: Self) -> Self {
        Self {
            val: self.val.mul_add(a.val, b.val),
            der: self.val.mul_add(a.der, self.der.mul_add(a.val, b.der)),
        }
    }
}

// ---------------------------------------------------------------------------
// DualN<T, K>: arithmetic
// ---------------------------------------------------------------------------

impl<T: Ring, const K: usize> Add for DualN<T, K> {
    type Output = Self;
    fn add(mut self, rhs: Self) -> Self {
        self.val += rhs.val;
        for (d, r) in self.der.iter_mut().zip(rhs.der.iter()) {
            *d += *r;
        }
        self
    }
}

impl<T: Ring, const K: usize> Sub for DualN<T, K> {
    type Output = Self;
    fn sub(mut self, rhs: Self) -> Self {
        self.val -= rhs.val;
        for (d, r) in self.der.iter_mut().zip(rhs.der.iter()) {
            *d -= *r;
        }
        self
    }
}

// The product rule, per derivative slot.
impl<T: Ring, const K: usize> Mul for DualN<T, K> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let mut der = self.der;
        for (d, r) in der.iter_mut().zip(rhs.der.iter()) {
            *d = self.val * *r + *d * rhs.val;
        }
        Self {
            val: self.val * rhs.val,
            der,
        }
    }
}

// The quotient rule; requires a nonzero standard part in `rhs`.
impl<T: Field, const K: usize> Div for DualN<T, K> {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        let val = self.val / rhs.val;
        let mut der = self.der;
        for (d, r) in der.iter_mut().zip(rhs.der.iter()) {
            *d = (*d - val * *r) / rhs.val;
        }
        Self { val, der }
    }
}

impl<T: Ring, const K: usize> Neg for DualN<T, K> {
    type Output = Self;
    fn neg(mut self) -> Self {
        self.val = -self.val;
        for d in self.der.iter_mut() {
            *d = -*d;
        }
        self
    }
}

impl<T: Ring, const K: usize> AddAssign for DualN<T, K> {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}
impl<T: Ring, const K: usize> SubAssign for DualN<T, K> {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}
impl<T: Ring, const K: usize> MulAssign for DualN<T, K> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}
impl<T: Field, const K: usize> DivAssign for DualN<T, K> {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl<T: Ring, const K: usize> Sum for DualN<T, K> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(
            Self {
                val: T::ZERO,
                der: [T::ZERO; K],
            },
            |a, b| a + b,
        )
    }
}

// Mixed-scalar ops, as for `Dual`.
impl<T: Ring, const K: usize> Add<T> for DualN<T, K> {
    type Output = Self;
    fn add(mut self, rhs: T) -> Self {
        self.val += rhs;
        self
    }
}
impl<T: Ring, const K: usize> Sub<T> for DualN<T, K> {
    type Output = Self;
    fn sub(mut self, rhs: T) -> Self {
        self.val -= rhs;
        self
    }
}
impl<T: Ring, const K: usize> Mul<T> for DualN<T, K> {
    type Output = Self;
    fn mul(mut self, rhs: T) -> Self {
        self.val *= rhs;
        for d in self.der.iter_mut() {
            *d *= rhs;
        }
        self
    }
}
impl<T: Field, const K: usize> Div<T> for DualN<T, K> {
    type Output = Self;
    fn div(mut self, rhs: T) -> Self {
        self.val /= rhs;
        for d in self.der.iter_mut() {
            *d /= rhs;
        }
        self
    }
}

impl<T: Ring, const K: usize> From<T> for DualN<T, K> {
    fn from(value: T) -> Self {
        Self::constant(value)
    }
}

// ---------------------------------------------------------------------------
// DualN<T, K>: tower + Algebra instances
// ---------------------------------------------------------------------------

impl<T: Ring, const K: usize> Element for DualN<T, K> {}
impl<T: Ring, const K: usize> Monoid for DualN<T, K> {
    const ZERO: Self = Self {
        val: T::ZERO,
        der: [T::ZERO; K],
    };
}
impl<T: Ring, const K: usize> Group for DualN<T, K> {}
impl<T: Ring, const K: usize> Semiring for DualN<T, K> {
    const ONE: Self = Self {
        val: T::ONE,
        der: [T::ZERO; K],
    };
}
impl<T: Ring, const K: usize> Ring for DualN<T, K> {}

impl<T: Field, const K: usize> Field for DualN<T, K> {
    fn recip(self) -> Self {
        let r = self.val.recip();
        let mut der = self.der;
        for d in der.iter_mut() {
            *d = -(*d * r * r);
        }
        Self { val: r, der }
    }
}

impl<T: Ring, const K: usize> Algebra<T> for DualN<T, K> {}

// Nested composition, as for `Dual`: real-coefficient systems evaluated at
// complex points differentiate against all K variables in one sweep (the
// Newton corrector's Jacobian for a real target system).
impl<F: Real, const K: usize> Add<F> for DualN<Complex<F>, K> {
    type Output = Self;
    fn add(mut self, rhs: F) -> Self {
        self.val = self.val + rhs;
        self
    }
}
impl<F: Real, const K: usize> Mul<F> for DualN<Complex<F>, K> {
    type Output = Self;
    fn mul(mut self, rhs: F) -> Self {
        self.val = self.val * rhs;
        for d in self.der.iter_mut() {
            *d = *d * rhs;
        }
        self
    }
}
impl<F: Real, const K: usize> From<F> for DualN<Complex<F>, K> {
    fn from(value: F) -> Self {
        Self::constant(Complex::from(value))
    }
}
impl<F: Real, const K: usize> Algebra<F> for DualN<Complex<F>, K> {}

// ---------------------------------------------------------------------------
// DualN<F: Real, K>: chain-rule lifts
// ---------------------------------------------------------------------------

/// Chain-rule lifts, as for [`Dual`]: the scalar factor from the chain rule
/// is applied to every derivative slot.
impl<F: Real, const K: usize> DualN<F, K> {
    /// Applies the already-computed chain factor to every slot.
    #[inline]
    fn lift(val: F, mut der: [F; K], factor: F) -> Self {
        for d in der.iter_mut() {
            *d *= factor;
        }
        Self { val, der }
    }

    /// `d(√x) = x′ / (2√x)`.
    pub fn sqrt(self) -> Self {
        let s = self.val.sqrt();
        Self::lift(s, self.der, (s + s).recip())
    }

    /// `d(sin x) = x′·cos x`.
    pub fn sin(self) -> Self {
        let (s, c) = self.val.sin_cos();
        Self::lift(s, self.der, c)
    }

    /// `d(cos x) = −x′·sin x`.
    pub fn cos(self) -> Self {
        let (s, c) = self.val.sin_cos();
        Self::lift(c, self.der, -s)
    }

    /// Both [`DualN::sin`] and [`DualN::cos`] from a single `sin_cos` call.
    pub fn sin_cos(self) -> (Self, Self) {
        let (s, c) = self.val.sin_cos();
        (Self::lift(s, self.der, c), Self::lift(c, self.der, -s))
    }

    /// `d(tan x) = x′·(1 + tan²x)`.
    pub fn tan(self) -> Self {
        let t = self.val.tan();
        Self::lift(t, self.der, F::ONE + t * t)
    }

    /// `d(eˣ) = x′·eˣ`.
    pub fn exp(self) -> Self {
        let e = self.val.exp();
        Self::lift(e, self.der, e)
    }

    /// `d(ln x) = x′ / x`.
    pub fn ln(self) -> Self {
        Self::lift(self.val.ln(), self.der, self.val.recip())
    }

    /// `d(xⁿ) = x′·n·xⁿ⁻¹`.
    pub fn powi(self, n: i32) -> Self {
        if n == 0 {
            return Self::constant(F::ONE);
        }
        let nf = if n < 0 {
            -F::from_u32(n.unsigned_abs())
        } else {
            F::from_u32(n as u32)
        };
        Self::lift(self.val.powi(n), self.der, nf * self.val.powi(n - 1))
    }

    /// `self·a + b` with the value part fused ([`Real::mul_add`]) and the
    /// derivative by the product rule.
    pub fn mul_add(self, a: Self, b: Self) -> Self {
        let mut der = self.der;
        for ((d, &da), &db) in der.iter_mut().zip(a.der.iter()).zip(b.der.iter()) {
            *d = self.val.mul_add(da, d.mul_add(a.val, db));
        }
        Self {
            val: self.val.mul_add(a.val, b.val),
            der,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::complex::c64;
    use crate::polynomial::Polynomial;

    fn d(val: f64, der: f64) -> Dual<f64> {
        Dual { val, der }
    }

    #[test]
    fn constructors() {
        assert_eq!(Dual::variable(3.0), d(3.0, 1.0));
        assert_eq!(Dual::constant(3.0), d(3.0, 0.0));

        let v = DualN::<f64, 3>::variable(2.0, 1);
        assert_eq!(v.val, 2.0);
        assert_eq!(v.der, [0.0, 1.0, 0.0]);
        assert_eq!(DualN::<f64, 3>::constant(2.0).der, [0.0; 3]);
    }

    #[test]
    #[should_panic]
    fn dualn_variable_rejects_out_of_range_slot() {
        let _ = DualN::<f64, 2>::variable(1.0, 2);
    }

    #[test]
    fn ring_identities() {
        let x = d(2.0, 3.0);
        let y = d(-1.0, 0.5);
        let z = d(4.0, -2.0);

        // Additive identities and inverses.
        assert_eq!(x + Dual::ZERO, x);
        assert_eq!(x + (-x), Dual::ZERO);
        assert_eq!(x - x, Dual::ZERO);

        // Multiplicative identity.
        assert_eq!(x * Dual::ONE, x);
        assert_eq!(Dual::<f64>::ONE * x, x);

        // Commutativity and associativity on exact dyadic values.
        assert_eq!(x + y, y + x);
        assert_eq!(x * y, y * x);
        assert_eq!((x + y) + z, x + (y + z));
        assert_eq!((x * y) * z, x * (y * z));

        // Distributivity.
        assert_eq!(x * (y + z), x * y + x * z);

        // ε² = 0: two pure-ε elements multiply to zero.
        let eps = d(0.0, 1.0);
        assert_eq!(eps * eps, Dual::ZERO);
    }

    #[test]
    fn field_identities() {
        let x = d(2.0, 3.0);
        // x · x⁻¹ = 1 exactly on powers of two.
        assert_eq!(x * Field::recip(x), Dual::ONE);
        // recip = (1/v, −d/v²).
        assert_eq!(Field::recip(x), d(0.5, -0.75));
        // Division against multiplication.
        let y = d(4.0, -2.0);
        assert_eq!((x * y) / y, x);
        // Quotient rule: (x/y)′ = (x′y − xy′)/y².
        let q = x / y;
        assert_eq!(q.val, 0.5);
        assert_eq!(q.der, (3.0 * 4.0 - 2.0 * (-2.0)) / 16.0);
    }

    #[test]
    fn mixed_scalar_ops() {
        let x = d(2.0, 3.0);
        assert_eq!(x + 1.0, d(3.0, 3.0));
        assert_eq!(x - 1.0, d(1.0, 3.0));
        assert_eq!(x * 2.0, d(4.0, 6.0));
        assert_eq!(x / 2.0, d(1.0, 1.5));

        let v = DualN::<f64, 2>::variable(2.0, 0);
        assert_eq!((v + 1.0).val, 3.0);
        assert_eq!((v * 3.0).der, [3.0, 0.0]);
        assert_eq!((v - 0.5).val, 1.5);
        assert_eq!((v / 2.0).der, [0.5, 0.0]);
    }

    #[test]
    fn sum_impls() {
        let total: Dual<f64> = [d(1.0, 2.0), d(3.0, 4.0), d(5.0, 6.0)]
            .into_iter()
            .sum();
        assert_eq!(total, d(9.0, 12.0));

        let vs = [
            DualN::<f64, 2>::variable(1.0, 0),
            DualN::<f64, 2>::variable(2.0, 1),
        ];
        let total: DualN<f64, 2> = vs.into_iter().sum();
        assert_eq!(total.val, 3.0);
        assert_eq!(total.der, [1.0, 1.0]);
    }

    #[test]
    fn eval_at_dual_matches_analytic_derivative() {
        // p(x) = x³ − 2x + 5 (ascending storage), p′(x) = 3x² − 2.
        let p = Polynomial::new([5.0, -2.0, 0.0, 1.0]);
        for x in [-2.0, -0.5, 0.0, 1.0, 3.0] {
            let r = p.eval_at(Dual::variable(x));
            assert_eq!(r.val, p.eval(x));
            assert_eq!(r.der, 3.0 * x * x - 2.0);
        }

        // At a constant the derivative slot stays zero.
        let r = p.eval_at(Dual::constant(2.0));
        assert_eq!(r.val, p.eval(2.0));
        assert_eq!(r.der, 0.0);
    }

    #[test]
    fn dualn_jacobian_of_two_variable_system() {
        // f₁(x, y) = x²y + y − 1,  f₂(x, y) = x + y³
        // J = [[2xy, x² + 1], [1, 3y²]]
        fn f<X: Copy + Ring + Mul<f64, Output = X> + Sub<f64, Output = X>>(x: X, y: X) -> [X; 2] {
            [x * x * y + y - 1.0, x + y * y * y]
        }

        let (x0, y0) = (2.0, -3.0);
        let x = DualN::<f64, 2>::variable(x0, 0);
        let y = DualN::<f64, 2>::variable(y0, 1);
        let [f1, f2] = f(x, y);

        assert_eq!(f1.val, x0 * x0 * y0 + y0 - 1.0);
        assert_eq!(f2.val, x0 + y0 * y0 * y0);
        assert_eq!(f1.der, [2.0 * x0 * y0, x0 * x0 + 1.0]);
        assert_eq!(f2.der, [1.0, 3.0 * y0 * y0]);
    }

    #[test]
    fn nested_dual_complex() {
        // h(t) = z·t² + w with complex z, w along a complex path t:
        // h′(t) = 2zt.
        let z = c64::new(1.0, 2.0);
        let w = c64::new(-3.0, 0.5);
        let t = Dual::variable(c64::new(0.5, -1.5));
        let h = t * t * Dual::constant(z) + Dual::constant(w);
        let t0 = c64::new(0.5, -1.5);
        assert_eq!(h.val, t0 * t0 * z + w);
        assert_eq!(h.der, (t0 + t0) * z);

        // Real coefficients at a Dual<Complex> point via Algebra<f64>:
        // p(x) = x² + 1 at x = i: value 0, derivative 2i.
        let p = Polynomial::new([1.0, 0.0, 1.0]);
        let r = p.eval_at(Dual::variable(c64::new(0.0, 1.0)));
        assert_eq!(r.val, c64::new(0.0, 0.0));
        assert_eq!(r.der, c64::new(0.0, 2.0));

        // And complex coefficients at a Dual<Complex> point via the generic
        // Algebra<Complex> instance.
        let q = Polynomial::new([w, z]);
        let s = q.eval_at(Dual::variable(t0));
        assert_eq!(s.val, z * t0 + w);
        assert_eq!(s.der, z);
    }

    #[test]
    fn dualn_nested_complex() {
        // Real-coefficient polynomial at a DualN<Complex> point.
        let p = Polynomial::new([1.0, 0.0, 1.0]); // x² + 1
        let x = DualN::<c64, 2>::variable(c64::new(0.0, 1.0), 1);
        let r = p.eval_at(x);
        assert_eq!(r.val, c64::new(0.0, 0.0));
        assert_eq!(r.der[0], c64::new(0.0, 0.0));
        assert_eq!(r.der[1], c64::new(0.0, 2.0));
    }

    #[test]
    fn chain_rule_lifts_dual() {
        let x = Dual::variable(0.7);

        let s = x.sqrt();
        assert!((s.val - 0.7_f64.sqrt()).abs() < 1e-15);
        assert!((s.der - 0.5 / 0.7_f64.sqrt()).abs() < 1e-15);

        let s = x.sin();
        assert!((s.val - 0.7_f64.sin()).abs() < 1e-15);
        assert!((s.der - 0.7_f64.cos()).abs() < 1e-15);

        let c = x.cos();
        assert!((c.val - 0.7_f64.cos()).abs() < 1e-15);
        assert!((c.der + 0.7_f64.sin()).abs() < 1e-15);

        let (s2, c2) = x.sin_cos();
        assert_eq!(s2, x.sin());
        assert_eq!(c2, x.cos());

        let t = x.tan();
        assert!((t.val - 0.7_f64.tan()).abs() < 1e-15);
        assert!((t.der - 1.0 / (0.7_f64.cos() * 0.7_f64.cos())).abs() < 1e-12);

        let e = x.exp();
        assert!((e.val - 0.7_f64.exp()).abs() < 1e-15);
        assert!((e.der - 0.7_f64.exp()).abs() < 1e-15);

        let l = x.ln();
        assert!((l.val - 0.7_f64.ln()).abs() < 1e-15);
        assert!((l.der - 1.0 / 0.7).abs() < 1e-15);

        // exp∘ln transports the derivative back to 1.
        let round = x.ln().exp();
        assert!((round.val - 0.7).abs() < 1e-15);
        assert!((round.der - 1.0).abs() < 1e-14);

        let p = x.powi(3);
        assert!((p.val - 0.343).abs() < 1e-15);
        assert!((p.der - 3.0 * 0.49).abs() < 1e-15);
        let p = x.powi(-2);
        assert!((p.der + 2.0 / 0.7_f64.powi(3)).abs() < 1e-12);
        assert_eq!(x.powi(0), Dual::constant(1.0));

        // mul_add == the composed ops, value and derivative.
        let a = d(2.0, -1.0);
        let b = d(0.25, 4.0);
        assert_eq!(x.mul_add(a, b), x * a + b);
    }

    #[test]
    fn chain_rule_lifts_dualn() {
        // g(x, y) = exp(x)·sin(y): ∂g/∂x = exp(x)sin(y), ∂g/∂y = exp(x)cos(y).
        let (x0, y0) = (0.3, 1.1);
        let x = DualN::<f64, 2>::variable(x0, 0);
        let y = DualN::<f64, 2>::variable(y0, 1);
        let g = x.exp() * y.sin();
        assert!((g.val - x0.exp() * y0.sin()).abs() < 1e-15);
        assert!((g.der[0] - x0.exp() * y0.sin()).abs() < 1e-15);
        assert!((g.der[1] - x0.exp() * y0.cos()).abs() < 1e-15);

        // sqrt/ln/tan/powi single-variable sanity through slot 0.
        let v = DualN::<f64, 1>::variable(0.7, 0);
        assert!((v.sqrt().der[0] - 0.5 / 0.7_f64.sqrt()).abs() < 1e-15);
        assert!((v.ln().der[0] - 1.0 / 0.7).abs() < 1e-15);
        assert!((v.tan().der[0] - 1.0 / (0.7_f64.cos() * 0.7_f64.cos())).abs() < 1e-12);
        assert!((v.powi(4).der[0] - 4.0 * 0.7_f64.powi(3)).abs() < 1e-15);
        assert_eq!(v.powi(0), DualN::constant(1.0));
        let (s, c) = v.sin_cos();
        assert_eq!(s, v.sin());
        assert_eq!(c, v.cos());

        let a = DualN::<f64, 1>::variable(2.0, 0);
        let b = DualN::<f64, 1>::constant(0.25);
        assert_eq!(v.mul_add(a, b), v * a + b);
    }

    #[test]
    fn dualn_field_identities() {
        let x = DualN::<f64, 2> {
            val: 2.0,
            der: [3.0, -1.0],
        };
        let r = Field::recip(x);
        assert_eq!(r.val, 0.5);
        assert_eq!(r.der, [-0.75, 0.25]);
        assert_eq!(x * r, DualN::ONE);

        let y = DualN::<f64, 2> {
            val: 4.0,
            der: [0.5, 2.0],
        };
        assert_eq!((x * y) / y, x);
        assert_eq!(x + (-x), DualN::ZERO);
        assert_eq!(x * DualN::ONE, x);
    }
}
