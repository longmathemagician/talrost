use core::ops::*;

use crate::element::Element;

pub trait Monoid: Element + Add<Output = Self> + AddAssign {
    const ZERO: Self;
}

#[macro_export]
macro_rules! impl_monoid {
    ($(($basis:ty, $additive_identity:expr)),+) => {
        $(
            impl Monoid for $basis {
                const ZERO: Self = $additive_identity;
            }
        )+
    };
}

pub trait Group: Monoid + Sub<Output = Self> + SubAssign + Neg<Output = Self> {
    #[allow(non_snake_case)]
    fn Neg(self) -> Self;
}

#[macro_export]
macro_rules! impl_group {
    ($(($basis: ty, $additive_identity: expr)),+) => {
        $(
            impl Monoid for $basis {
                const ZERO: Self = $additive_identity;
            }
            impl Group for $basis {
                fn Neg(self) -> Self {
                    -self
                }
            }
        )+
    };
}

pub trait Semiring: Monoid + Mul<Output = Self> + MulAssign {
    const ONE: Self;
}

#[macro_export]
macro_rules! impl_semiring {
    ($(($basis: ty, $multiplicative_identity: expr)),+) => {
        $(
            impl Semiring for $basis {
                const ONE: Self = $multiplicative_identity;
            }
        )+
    };
}

pub trait Ring: Group + Semiring {}
#[macro_export]
macro_rules! impl_ring {
    ($($base_type: ty),+) => {
        $(
            impl Ring for $base_type {}
        )+
    };
}

pub trait Field: Ring + Div<Output = Self> + DivAssign {
    #[allow(non_snake_case)]
    fn recip(self) -> Self;
}
#[macro_export]
macro_rules! impl_field {
    ($($base_type: ty),+) => {
        $(
            impl Field for $base_type {
                #[allow(unconditional_recursion)]
                fn recip(self) -> Self {
                    Self::recip(self)
                }
            }
        )+
    };
}
