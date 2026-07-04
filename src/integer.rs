//! The [`Integer`] trait: signed machine integers as ordered rings.

use crate::algebra::*;
use crate::element::Element;
use crate::{impl_group, impl_monoid, impl_ring, impl_semiring};

/// The signed machine-integer family: `i8`, `i16`, `i32`, `i64`, `i128`,
/// `isize`. Floats do *not* implement this trait; they live under
/// [`crate::real::Real`].
pub trait Integer: Ring + Ord + Eq {
    /// The smallest representable value (`i32::MIN`-style).
    const MIN: Self;
    /// The largest representable value (`i32::MAX`-style).
    const MAX: Self;
    /// Bit width of the type (`i32::BITS`-style), as a `u32`.
    const BITS: u32;

    /// Raises `self` to a non-negative integer power. Negative exponents are
    /// unrepresentable here by construction — no runtime assert needed.
    fn pow(self, exp: u32) -> Self;

    /// The absolute value. Panics (or wraps, per the build's overflow
    /// semantics) on `MIN`, whose magnitude is unrepresentable.
    fn abs(self) -> Self;
}

macro_rules! stack_integer {
    ($($T:ty),+) => {
        $(
            impl Element for $T {}
            impl_monoid!(($T, 0));
            impl_group!($T);
            impl_semiring!(($T, 1));
            impl_ring!($T);

            impl Integer for $T {
                const MIN: Self = <$T>::MIN;
                const MAX: Self = <$T>::MAX;
                const BITS: u32 = <$T>::BITS;

                fn pow(self, exp: u32) -> Self {
                    <$T>::pow(self, exp)
                }

                fn abs(self) -> Self {
                    <$T>::abs(self)
                }
            }
        )+
    };
}

stack_integer!(i8, i16, i32, i64, i128, isize);

#[cfg(test)]
mod tests {
    use super::*;

    fn neg<T: Integer>(a: T, b: T) {
        assert_eq!(-a, b);
    }

    #[test]
    fn integer_neg() {
        neg(1_i32, -1_i32);
        neg(-7_i64, 7_i64);
    }

    #[test]
    fn integer_pow() {
        // Ported from the deleted `Number` trait tests.
        fn pow2<T: Integer>(n: T, answer: T) {
            assert_eq!(n.pow(2), answer);
        }
        pow2(64_i32, 4096_i32);
        pow2(128_i64, 16384_i64);
        pow2(-3_i8, 9_i8);

        assert_eq!(2_i32.pow(0), 1);
    }

    #[test]
    fn integer_abs() {
        fn abs<T: Integer>(a: T, b: T) {
            assert_eq!(a.abs(), b);
        }
        abs(-5_i32, 5_i32);
        abs(5_i32, 5_i32);
        abs(-128_i16, 128_i16);
        abs(0_isize, 0_isize);
    }

    #[test]
    fn integer_consts() {
        assert_eq!(<i8 as Integer>::BITS, 8);
        assert_eq!(<i64 as Integer>::BITS, 64);
        assert_eq!(<i8 as Integer>::MIN, -128);
        assert_eq!(<i8 as Integer>::MAX, 127);
    }
}
