#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(feature = "specialization", feature(min_specialization))]

// The `Real` math functions (sqrt, sin, fma, ...) need a backend: either the
// standard library (default) or the `libm` crate for no_std targets.
#[cfg(not(any(feature = "std", feature = "libm")))]
compile_error!(
    "talrost requires a float math backend: enable the `std` feature (default) or `libm`."
);

// The algebraic tower.
pub mod algebra;
pub mod element;

// Numeric families: unsigned/signed machine integers, IEEE floats, and the
// Scalar abstraction (field with a real-valued norm) over floats and complex.
pub mod complex;
pub mod integer;
pub mod natural;
pub mod real;
pub mod scalar;

// Forward-mode automatic differentiation as ring elements.
pub mod dual;

// Containers and solvers.
pub mod lattice;
pub mod matrix;
pub mod mvpoly;
pub mod polynomial;
pub mod roots;
pub mod solvers;
pub mod vector;
