use core::ops::{Div, Mul};

use crate::algebra::Field;
use crate::complex::Complex;
use crate::real::Real;

/// What `Vector`/`Matrix`/`Polynomial` actually need: a field with a
/// real-valued norm.
///
/// Implemented by every [`Real`] type (with `Real = Self`) and by
/// [`Complex<F>`] (with `Real = F`). The `Mul`/`Div`-by-`Self::Real`
/// supertraits let generic code scale a scalar by a real quantity — e.g.
/// dividing a complex vector component by its (real) magnitude.
pub trait Scalar: Field + Mul<Self::Real, Output = Self> + Div<Self::Real, Output = Self> {
    type Real: Real;

    /// Squared norm, `|x|²`: cheap and exact (no square root).
    fn norm_sqr(self) -> Self::Real;

    /// Norm (absolute value / complex modulus), a *real* number.
    fn norm(self) -> Self::Real {
        self.norm_sqr().sqrt()
    }

    /// Complex conjugate; the identity for real types.
    fn conj(self) -> Self;

    /// `self * a + b`. Overridden with the fused operation for real types so
    /// Horner evaluation gets FMA where the type supports it.
    fn mul_add(self, a: Self, b: Self) -> Self {
        self * a + b
    }

    fn is_nan(self) -> bool;
    fn is_finite(self) -> bool;
}

// Coherent with the `Complex` impl below because `Complex` does not (and must
// not) implement `Real`.
impl<F: Real> Scalar for F {
    type Real = F;

    fn norm_sqr(self) -> F {
        self * self
    }

    fn norm(self) -> F {
        self.abs()
    }

    fn conj(self) -> F {
        self
    }

    fn mul_add(self, a: F, b: F) -> F {
        Real::mul_add(self, a, b)
    }

    fn is_nan(self) -> bool {
        Real::is_nan(self)
    }

    fn is_finite(self) -> bool {
        Real::is_finite(self)
    }
}

impl<F: Real> Scalar for Complex<F> {
    type Real = F;

    fn norm_sqr(self) -> F {
        self.re * self.re + self.im * self.im
    }

    fn conj(self) -> Self {
        Self::new(self.re, -self.im)
    }

    fn is_nan(self) -> bool {
        Real::is_nan(self.re) || Real::is_nan(self.im)
    }

    fn is_finite(self) -> bool {
        Real::is_finite(self.re) && Real::is_finite(self.im)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::complex::c64;

    #[test]
    fn scalar_real_f64() {
        let x = -3.0_f64;
        assert_eq!(Scalar::norm_sqr(x), 9.0);
        assert_eq!(Scalar::norm(x), 3.0);
        assert_eq!(Scalar::conj(x), -3.0);
        assert_eq!(Scalar::mul_add(2.0, 3.0, 4.0), 10.0);
        assert!(Scalar::is_nan(f64::NAN));
        assert!(!Scalar::is_nan(1.0_f64));
        assert!(Scalar::is_finite(1.0_f64));
        assert!(!Scalar::is_finite(f64::INFINITY));
    }

    #[test]
    fn scalar_complex_c64() {
        let z = c64::new(3.0, 4.0);
        // norm_sqr is a real (f64) value, not a complex one.
        let n: f64 = z.norm_sqr();
        assert_eq!(n, 25.0);
        let m: f64 = z.norm();
        assert_eq!(m, 5.0);
        assert_eq!(z.conj(), c64::new(3.0, -4.0));

        // Default mul_add: z * a + b, checked against direct arithmetic.
        let a = c64::new(1.0, 2.0);
        let b = c64::new(-5.0, 0.5);
        assert_eq!(Scalar::mul_add(z, a, b), z * a + b);

        assert!(!Scalar::is_nan(z));
        assert!(Scalar::is_nan(c64::new(f64::NAN, 0.0)));
        assert!(Scalar::is_finite(z));
        assert!(!Scalar::is_finite(c64::new(f64::INFINITY, 0.0)));
    }

    #[test]
    fn scalar_scaling_by_real() {
        // The Mul/Div<Self::Real> supertraits at work, generically.
        fn scale<T: Scalar>(x: T, s: T::Real) -> T {
            x * s
        }
        fn unscale<T: Scalar>(x: T, s: T::Real) -> T {
            x / s
        }
        assert_eq!(scale(3.0_f64, 2.0), 6.0);
        assert_eq!(unscale(3.0_f64, 2.0), 1.5);
        assert_eq!(scale(c64::new(1.0, -2.0), 2.0), c64::new(2.0, -4.0));
        assert_eq!(unscale(c64::new(2.0, -4.0), 2.0), c64::new(1.0, -2.0));
    }
}
