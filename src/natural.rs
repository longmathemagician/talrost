use crate::algebra::*;
use crate::element::Element;
use crate::{impl_monoid, impl_semiring};

pub trait Natural: Semiring + PartialOrd + PartialEq {
    const MIN: Self;
    const MAX: Self;
    const BITS: Self;

    fn floor(&self) -> Self;
    fn ceil(&self) -> Self;
    fn abs(&self) -> Self;
    fn powi(&self, power: i32) -> Self;
    fn sin(&self) -> Self;
    fn cos(&self) -> Self;
    fn tan(&self) -> Self;
    fn atan2(&self, other: Self) -> Self;
}

macro_rules! stack_natural {
    ($($T:ty),+) => {
        $(
            impl Element for $T {}
            impl_monoid!(($T, 0));
            impl_semiring!(($T, 1));

            impl Natural for $T {
                const MIN: Self = Self::MIN;
                const MAX: Self = Self::MAX;
                const BITS: Self = Self::BITS as Self;

                #[allow(unconditional_recursion)]
                fn floor(&self) -> Self {
                    Self::floor(&self)
                }
                #[allow(unconditional_recursion)]
                fn ceil(&self) -> Self {
                    Self::ceil(&self)
                }
                #[allow(unconditional_recursion)]
                fn abs(&self) -> Self {
                    Self::abs(&self)
                }
                fn powi(&self, power: i32) -> Self {
                    Self::pow(*self, power as u32)
                }
                #[allow(unconditional_recursion)]
                fn sin(&self) -> Self {
                    Self::sin(&self)
                }
                #[allow(unconditional_recursion)]
                fn cos(&self) -> Self {
                    Self::cos(&self)
                }
                #[allow(unconditional_recursion)]
                fn tan(&self) -> Self {
                    Self::tan(&self)
                }
                #[allow(unconditional_recursion)]
                fn atan2(&self, other: Self) -> Self {
                    Self::atan2(&self, other)
                }
            }
        )+
    };
}

stack_natural!(u8, u16, u32, u64, u128, usize);

#[cfg(test)]
mod tests {
    use super::*;

    // Helper function to test the inherited natural trait methods
    fn test_natural_trait_methods<T: Natural>(a: T) {
        assert_eq!(a.floor(), a);
        assert_eq!(a.ceil(), a);
        assert_eq!(a.abs(), a);
        assert_eq!(a.powi(0), T::ONE);
    }

    #[test]
    // Main test function
    fn test_float_trait() {
        let a: f64 = 1.0;
        // test_inverse(a, -a);
        test_natural_trait_methods(a);
        // test_integer_trait_methods(a);
    }
}
