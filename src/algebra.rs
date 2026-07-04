use core::ops::*;

use crate::element::Element;

/// Additive monoid: a set closed under an associative `+` with an identity
/// element [`Monoid::ZERO`]. The `+` convention is additive throughout this
/// crate; `core::iter::Sum` is required so generic code can `.sum()`.
pub trait Monoid: Element + Add<Output = Self> + AddAssign + core::iter::Sum {
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

/// Additive group: a [`Monoid`] where every element has an additive inverse
/// (`-x` / `x - y` via the `Neg`/`Sub` supertraits).
pub trait Group: Monoid + Sub<Output = Self> + SubAssign + Neg<Output = Self> {}

#[macro_export]
macro_rules! impl_group {
    ($($basis:ty),+) => {
        $(
            impl Group for $basis {}
        )+
    };
}

/// Semiring: an additive [`Monoid`] that is also closed under an associative
/// `*` with identity [`Semiring::ONE`].
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
    fn recip(self) -> Self;
}
#[macro_export]
macro_rules! impl_field {
    ($($base_type: ty),+) => {
        $(
            impl Field for $base_type {
                fn recip(self) -> Self {
                    // Call the type's inherent `recip` unambiguously; `Self::recip`
                    // resolves back to this trait method when no inherent one exists.
                    <$base_type>::recip(self)
                }
            }
        )+
    };
}
