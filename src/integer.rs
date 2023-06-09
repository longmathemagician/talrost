use crate::algebra::*;
use crate::element::Element;
use crate::natural::Natural;
use crate::{impl_group, impl_ring, impl_semiring};

macro_rules! impl_natural_for_integer {
    ($($base_type: ty),+) => {
        $(
            impl Natural for $base_type {
                const MIN: Self = Self::MIN;
                const MAX: Self = Self::MAX;
                const BITS: Self = Self::BITS as Self;

                // fn abs(&self) -> Self {
                //     Self::abs(*self)
                // }
                fn powi(&self, power: i32) -> Self {
                    Self::pow(*self, power as u32)
                }
            }
        )+
    };
}

pub trait Integer: Natural + Ring {}

macro_rules! stack_integer{
    ($($base_type: ty),+) => {
        $(
            impl Element for $base_type {}
            impl_group!(($base_type, 0));
            impl_semiring!(($base_type, 1));
            impl_ring!($base_type);

            impl_natural_for_integer!($base_type);

            impl Integer for $base_type {}
        )+
    };
}

stack_integer!(i8, i16, i32, i64, i128, isize);

#[cfg(test)]
mod tests {
    use super::*;

    fn test_neg<T: Integer>(a: T, b: T) {
        assert_eq!(a.neg(), b);
    }

    #[test]
    fn test_ints() {
        let a: i32 = 1;
        let b: i32 = -1;
        test_neg(a, b);
    }
}
