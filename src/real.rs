use crate::algebra::*;
use crate::element::Element;
use crate::{impl_field, impl_group, impl_monoid, impl_ring, impl_semiring};

/// An ordered field with IEEE-754 float semantics: `f32` and `f64` (later
/// `f16`/`f128`).
///
/// This is the "real number" layer of the tower. Complex numbers deliberately
/// do **not** implement `Real` (there is no meaningful order on ℂ); generic
/// containers that only need a field with a real-valued norm should bound on
/// [`crate::scalar::Scalar`] instead.
pub trait Real: Field + PartialOrd {
    const EPSILON: Self;
    const INFINITY: Self;
    const NEG_INFINITY: Self;
    const NAN: Self;
    const MIN: Self;
    const MAX: Self;

    const DIGITS: u32;
    const MANTISSA_DIGITS: u32;
    const RADIX: u32;

    const MIN_EXP: i32;
    const MAX_EXP: i32;

    /// Builds a small constant from an unsigned integer (`n as f64`), so
    /// generic solver code can write `T::from_u32(3)` instead of
    /// `T::ONE + T::ONE + T::ONE`.
    fn from_u32(n: u32) -> Self;

    fn abs(self) -> Self;
    fn floor(self) -> Self;
    fn ceil(self) -> Self;

    fn sqrt(self) -> Self;
    fn cbrt(self) -> Self;

    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn tan(self) -> Self;
    fn sin_cos(self) -> (Self, Self);
    fn atan2(self, other: Self) -> Self;

    fn mul_add(self, a: Self, b: Self) -> Self;
    fn copysign(self, sign: Self) -> Self;
    fn powi(self, n: i32) -> Self;

    fn is_nan(self) -> bool;
    fn is_finite(self) -> bool;
}

macro_rules! stack_real {
    ($($basis: ty),+) => {
        $(
            impl Element for $basis {}
            impl_monoid!(($basis, 0.0));
            impl_group!($basis);
            impl_semiring!(($basis, 1.0));
            impl_ring!($basis);
            impl_field!($basis);

            impl Real for $basis {
                const EPSILON: Self = <$basis>::EPSILON;
                const INFINITY: Self = <$basis>::INFINITY;
                const NEG_INFINITY: Self = <$basis>::NEG_INFINITY;
                const NAN: Self = <$basis>::NAN;
                const MIN: Self = <$basis>::MIN;
                const MAX: Self = <$basis>::MAX;

                const DIGITS: u32 = <$basis>::DIGITS;
                const MANTISSA_DIGITS: u32 = <$basis>::MANTISSA_DIGITS;
                const RADIX: u32 = <$basis>::RADIX;

                const MIN_EXP: i32 = <$basis>::MIN_EXP;
                const MAX_EXP: i32 = <$basis>::MAX_EXP;

                fn from_u32(n: u32) -> Self {
                    n as $basis
                }

                fn abs(self) -> Self {
                    <$basis>::abs(self)
                }
                fn floor(self) -> Self {
                    <$basis>::floor(self)
                }
                fn ceil(self) -> Self {
                    <$basis>::ceil(self)
                }

                fn sqrt(self) -> Self {
                    <$basis>::sqrt(self)
                }
                fn cbrt(self) -> Self {
                    <$basis>::cbrt(self)
                }

                fn sin(self) -> Self {
                    <$basis>::sin(self)
                }
                fn cos(self) -> Self {
                    <$basis>::cos(self)
                }
                fn tan(self) -> Self {
                    <$basis>::tan(self)
                }
                fn sin_cos(self) -> (Self, Self) {
                    <$basis>::sin_cos(self)
                }
                fn atan2(self, other: Self) -> Self {
                    <$basis>::atan2(self, other)
                }

                fn mul_add(self, a: Self, b: Self) -> Self {
                    <$basis>::mul_add(self, a, b)
                }
                fn copysign(self, sign: Self) -> Self {
                    <$basis>::copysign(self, sign)
                }
                fn powi(self, n: i32) -> Self {
                    <$basis>::powi(self, n)
                }

                fn is_nan(self) -> bool {
                    <$basis>::is_nan(self)
                }
                fn is_finite(self) -> bool {
                    <$basis>::is_finite(self)
                }
            }
        )+
    };
}

stack_real!(f32, f64);

#[cfg(test)]
mod tests {
    use super::*;

    fn rounding_methods<T: Real>(a: T) {
        assert_eq!(a.floor(), a);
        assert_eq!(a.ceil(), a);
        assert_eq!(a.abs(), a);
        assert_eq!(a.powi(0), T::ONE);
    }

    fn inverse<T: Real>(a: T, b: T) {
        assert_eq!(-a, b);
    }

    fn sqrt<T: Real>(a: T, b: T) {
        assert_eq!(a.sqrt(), b);
    }

    #[test]
    fn real_trait_methods() {
        let a: f64 = 1.0;
        inverse(a, -1.0);
        rounding_methods(a);

        let b: f32 = 256.0;
        sqrt(b, 16.0);
    }

    #[test]
    fn real_from_u32() {
        assert_eq!(f64::from_u32(0), 0.0);
        assert_eq!(f64::from_u32(3), 3.0);
        assert_eq!(f32::from_u32(7), 7.0);
        assert_eq!(f64::from_u32(4_000_000_000), 4.0e9);
    }

    #[test]
    fn real_powi() {
        // Ported from the deleted `Number` trait tests.
        fn pow2<T: Real>(n: T, answer: T) {
            assert_eq!(n.powi(2), answer);
        }
        pow2(256_f32, 65536_f32);
        pow2(512_f64, 262144_f64);

        // Negative exponents are fine in a field.
        assert_eq!(2_f64.powi(-2), 0.25);
    }

    #[test]
    fn real_consts() {
        assert_eq!(<f64 as Real>::MANTISSA_DIGITS, 53);
        assert_eq!(<f32 as Real>::MANTISSA_DIGITS, 24);
        assert_eq!(<f64 as Real>::RADIX, 2);
        assert!(<f64 as Real>::NAN.is_nan());
        assert!(!<f64 as Real>::INFINITY.is_finite());
        assert!(<f64 as Real>::NEG_INFINITY < <f64 as Real>::MIN);
        assert!(<f64 as Real>::MAX < <f64 as Real>::INFINITY);
        assert_eq!(<f64 as Real>::EPSILON, f64::EPSILON);
    }
}
