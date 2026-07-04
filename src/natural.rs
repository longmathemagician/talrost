use crate::algebra::*;
use crate::element::Element;
use crate::{impl_monoid, impl_semiring};

/// The unsigned machine-integer family: `u8`, `u16`, `u32`, `u64`, `u128`,
/// `usize`. Floats and complex numbers do *not* implement this trait; they
/// live under [`crate::real::Real`] and [`crate::scalar::Scalar`].
pub trait Natural: Semiring + Ord + Eq {
    const MIN: Self;
    const MAX: Self;
    /// Bit width of the type (`u32::BITS`-style), as a `u32`.
    const BITS: u32;

    /// Raises `self` to a non-negative integer power. Negative exponents are
    /// unrepresentable here by construction — no runtime assert needed.
    fn pow(self, exp: u32) -> Self;
}

macro_rules! stack_natural {
    ($($T:ty),+) => {
        $(
            impl Element for $T {}
            impl_monoid!(($T, 0));
            impl_semiring!(($T, 1));

            impl Natural for $T {
                const MIN: Self = <$T>::MIN;
                const MAX: Self = <$T>::MAX;
                const BITS: u32 = <$T>::BITS;

                fn pow(self, exp: u32) -> Self {
                    <$T>::pow(self, exp)
                }
            }
        )+
    };
}

stack_natural!(u8, u16, u32, u64, u128, usize);

#[cfg(test)]
mod tests {
    use super::*;

    fn pow2<T: Natural>(n: T, answer: T) {
        assert_eq!(n.pow(2), answer);
    }

    #[test]
    fn natural_pow() {
        // Ported from the deleted `Number` trait tests.
        pow2(16_u32, 256_u32);
        pow2(32_u64, 1024_u64);
        pow2(3_u8, 9_u8);

        fn pow0<T: Natural>(n: T) {
            assert_eq!(n.pow(0), T::ONE);
        }
        pow0(7_u16);
        pow0(7_usize);
        pow0(7_u128);
    }

    #[test]
    fn natural_consts() {
        assert_eq!(<u8 as Natural>::BITS, 8);
        assert_eq!(<u32 as Natural>::BITS, 32);
        assert_eq!(<u128 as Natural>::BITS, 128);
        assert_eq!(<u8 as Natural>::MIN, 0);
        assert_eq!(<u8 as Natural>::MAX, 255);
    }
}
