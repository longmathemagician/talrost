//! The algebraic tower: `Monoid` → `Group` → `Semiring` → `Ring` → `Field`,
//! plus [`Algebra<T>`], the crate's single evaluation abstraction.
//!
//! Numeric families implement prefixes of the tower (unsigned integers stop
//! at [`Semiring`], signed integers at [`Ring`]; floats, complex numbers,
//! and duals reach [`Field`]), and generic code bounds on the weakest
//! structure it needs — matrix multiplication needs only `Ring`, so integer
//! exponent matrices work; LU solving needs a field with a real-valued norm
//! ([`crate::scalar::Scalar`]).
//!
//! The `impl_*` macros stamp tower impls onto concrete types, keeping the
//! per-type boilerplate in one place.

use core::ops::*;

use crate::element::Element;

/// Additive monoid: a set closed under an associative `+` with an identity
/// element [`Monoid::ZERO`]. The `+` convention is additive throughout this
/// crate; `core::iter::Sum` is required so generic code can `.sum()`.
pub trait Monoid: Element + Add<Output = Self> + AddAssign + core::iter::Sum {
    /// The additive identity: `x + ZERO == x`.
    const ZERO: Self;
}

/// Implements [`Monoid`] for one or more types, given each type's additive
/// identity: `impl_monoid!((u32, 0), (f64, 0.0))`.
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

/// Implements the marker trait [`Group`] for one or more types (which must
/// already be `Monoid + Sub + SubAssign + Neg`): `impl_group!(i32, f64)`.
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
    /// The multiplicative identity: `x * ONE == x`.
    const ONE: Self;
}

/// Implements [`Semiring`] for one or more types, given each type's
/// multiplicative identity: `impl_semiring!((u32, 1), (f64, 1.0))`.
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

/// Ring: a [`Group`] under `+` that is also a [`Semiring`] under `*` (with
/// multiplication distributing over addition). Signed integers stop here;
/// this is the bound for the structural container operations (matrix and
/// polynomial arithmetic without division).
pub trait Ring: Group + Semiring {}

/// Implements the marker trait [`Ring`] for one or more types that are
/// already `Group + Semiring`: `impl_ring!(i32, f64)`.
#[macro_export]
macro_rules! impl_ring {
    ($($base_type: ty),+) => {
        $(
            impl Ring for $base_type {}
        )+
    };
}

/// Field: a [`Ring`] whose nonzero elements have multiplicative inverses
/// (`/` via the `Div`/`DivAssign` supertraits, [`Field::recip`] directly).
///
/// The float/complex/dual implementations are fields in the IEEE-pragmatic
/// sense: division by zero yields infinities or NaNs rather than being
/// undefined.
pub trait Field: Ring + Div<Output = Self> + DivAssign {
    /// The multiplicative inverse, `1/self`.
    fn recip(self) -> Self;
}

/// An (associative, unital) algebra over the ring `T`: a ring `X` that can
/// absorb `T` on the right of `*`/`+` and be built from a `T`.
///
/// This is the crate's single evaluation abstraction: a polynomial with
/// coefficients in `T` can be evaluated at any point of any `Algebra<T>` with
/// one generic Horner/term loop. Instances:
///
/// - every ring over itself (plain evaluation, `X = T`);
/// - `Complex<F>` over `F` (real coefficients at complex points — Aberth /
///   Durand–Kerner, the γ-trick);
/// - `Dual<T>` / `DualN<T, K>` over `T` (derivatives and gradients by forward
///   automatic differentiation);
/// - `Dual<Complex<F>>` over both `Complex<F>` and `F` (derivatives of complex
///   paths with real coefficients).
pub trait Algebra<T: Ring>: Ring + Mul<T, Output = Self> + Add<T, Output = Self> + From<T> {}

// Every ring is an algebra over itself. (No overlap with the concrete
// instances elsewhere: e.g. this blanket gives `Complex<F>: Algebra<Complex<F>>`
// while `complex.rs` gives `Complex<F>: Algebra<F>` — different trait
// parameterizations.)
impl<T: Ring> Algebra<T> for T {}

/// Implements [`Field`] for one or more types by delegating to each type's
/// inherent `recip` method: `impl_field!(f32, f64)`.
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
