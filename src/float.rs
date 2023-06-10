use crate::algebra::*;
use crate::element::Element;
use crate::integer::Integer;
use crate::natural::Natural;
use crate::{impl_field, impl_group, impl_ring, impl_semiring};

macro_rules! impl_natural_for_float {
    ($($base_type: ty),+) => {
        $(
            impl Natural for $base_type {
                const MIN: Self = Self::MIN;
                const MAX: Self = Self::MAX;
                const BITS: Self = Self::ZERO.to_bits() as Self; // Totally wrong but leave it for now

                #[allow(unconditional_recursion)]
                fn floor(&self) -> Self {
                    Self::floor(*self)
                }
                #[allow(unconditional_recursion)]
                fn ceil(&self) -> Self {
                    Self::ceil(*self)
                }
                #[allow(unconditional_recursion)]
                fn abs(&self) -> Self {
                    Self::abs(*self)
                }
                fn powi(&self, power: i32) -> Self {
                    Self::powi(*self, power)
                }
                #[allow(unconditional_recursion)]
                fn sin(&self) -> Self {
                    Self::sin(*self)
                }
                #[allow(unconditional_recursion)]
                fn cos(&self) -> Self {
                    Self::cos(*self)
                }
                #[allow(unconditional_recursion)]
                fn tan(&self) -> Self {
                    Self::tan(*self)
                }
                #[allow(unconditional_recursion)]
                fn atan2(&self, other: Self) -> Self {
                    Self::atan2(*self, other)
                }
            }
        )+
    };
}
pub trait Float: Natural + Field {
    const DIGITS: u32;
    const MANTISSA_DIGITS: u32;
    const RADIX: u32;
    const MIN_EXP: i32;
    const MAX_EXP: i32;
    const INFINITY: Self;
    const NEG_INFINITY: Self;
    const NAN: Self;
    const EPSILON: Self;

    fn sqrt(&self) -> Self;
}

macro_rules! stack_float{
    ($($basis: ty),+) => {
        $(
            impl Element for $basis {}
            impl_group!(($basis, 0.0));
            impl_semiring!(($basis, 1.0));
            impl_ring!($basis);
            impl_field!($basis);

            impl_natural_for_float!($basis);
            impl Integer for $basis {}

            impl Float for $basis {
                const DIGITS: u32 = Self::DIGITS;
                const MANTISSA_DIGITS: u32 = Self::MANTISSA_DIGITS;
                const RADIX: u32 = Self::RADIX;
                const MIN_EXP: i32 = Self::MIN_EXP;
                const MAX_EXP: i32 = Self::MAX_EXP;
                const INFINITY: Self = Self::INFINITY;
                const NEG_INFINITY: Self = Self::NEG_INFINITY;
                const NAN: Self = Self::NAN;
                const EPSILON: Self = Self::EPSILON;

                fn sqrt(&self) -> Self {
                    <$basis>::sqrt(*self)
                }
            }
        )+
    };
}

stack_float!(f32, f64);

#[cfg(test)]
mod tests {
    use super::*;

    // Helper function to test the inherited natural trait methods
    fn test_natural_trait_methods<T: Float>(a: T) {
        assert_eq!(a.floor(), a);
        assert_eq!(a.ceil(), a);
        assert_eq!(a.abs(), a);
        assert_eq!(a.powi(0), T::ONE);
    }

    fn test_inverse<T: Float>(a: T, b: T) {
        assert_eq!(a.neg(), b);
    }

    // Helper function to test the inherited integer trait methods
    fn test_integer_trait_methods<T: Float>(a: T) {
        assert_eq!(a.neg(), -a);
        // assert_eq!(a.recip(), a);
    }

    fn test_sqrt<T: Float>(a: T, b: T) {
        assert_eq!(a.sqrt(), b);
    }

    #[test]
    // Main test function
    fn test_float_trait() {
        let a: f64 = 1.0;
        test_inverse(a, -a);
        test_natural_trait_methods(a);
        test_integer_trait_methods(a);

        let b: f32 = 256.0;
        test_sqrt(b, 16.0);
    }
}
