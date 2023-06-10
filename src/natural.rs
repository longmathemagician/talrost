use crate::algebra::*;
use crate::element::Element;
use crate::{impl_monoid, impl_semiring};

pub trait Natural: Semiring + PartialOrd + PartialEq {
    const MIN: Self;
    const MAX: Self;
    const BITS: Self;

    fn powi(&self, power: i32) -> Self;
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

                fn powi(&self, power: i32) -> Self {
                    Self::pow(*self, power as u32)
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
        assert_eq!(a.powi(0), T::ONE);
    }

    #[test]
    // Main test function
    fn test_natural_trait() {
        let a: f64 = 1.0;
        test_natural_trait_methods(a);
    }
}
