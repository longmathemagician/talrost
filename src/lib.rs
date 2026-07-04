//! # talrost
//!
//! A stack-only, allocation-free mathematics library built toward a
//! **polyhedral homotopy continuation solver**: every type is `Copy`, every
//! size is a const generic, and every generic routine monomorphizes to
//! straight-line code that also runs on `no_std` embedded targets.
//!
//! The layers, bottom up:
//!
//! - an algebraic tower of traits ([`algebra`]: `Monoid` → `Group` →
//!   `Semiring` → `Ring` → `Field`) implemented by machine integers
//!   ([`natural`], [`integer`]), IEEE floats ([`real`]), complex numbers
//!   ([`complex`]), and dual numbers ([`dual`]);
//! - [`algebra::Algebra<T>`], the single evaluation abstraction: one Horner
//!   or term loop evaluates a `T`-coefficient polynomial at points of any
//!   algebra over `T` — plain values, complex points, or dual numbers
//!   (forward-mode automatic differentiation);
//! - containers and solvers: [`polynomial`] (with root finders in
//!   [`solvers`]), [`vector`], [`matrix`] (LU-based solve/determinant/
//!   inverse), [`mvpoly`] (sparse multivariate systems with one-sweep
//!   Jacobians), and [`lattice`] (Smith/Hermite normal forms of integer
//!   matrices).
//!
//! # Example: the homotopy-tracking primitives
//!
//! Evaluate a real polynomial at a complex point, then Newton-correct a
//! 2-variable polynomial system using its forward-AD Jacobian and an LU
//! solve — the inner loop of a path tracker:
//!
//! ```
//! use talrost::complex::c64;
//! use talrost::mvpoly::{MPoly, MSystem, Monomial};
//! use talrost::polynomial::Polynomial;
//! use talrost::vector::Vector;
//!
//! // p(x) = x² + 1 (ascending storage: c[i] multiplies x^i) vanishes at i.
//! let p = Polynomial::new([1.0, 0.0, 1.0]);
//! assert_eq!(p.eval_at(c64::new(0.0, 1.0)), c64::new(0.0, 0.0));
//!
//! // f1 = x² + y² − 5, f2 = xy − 2, with a root at (2, 1).
//! let f1 = MPoly::<f64, 2, 3>::new(
//!     [1.0, 1.0, -5.0],
//!     [Monomial::new([2, 0]), Monomial::new([0, 2]), Monomial::new([0, 0])],
//! );
//! let f2 = MPoly::<f64, 2, 3>::new(
//!     [1.0, -2.0, 0.0],
//!     [Monomial::new([1, 1]), Monomial::new([0, 0]), Monomial::new([0, 0])],
//! );
//! let sys = MSystem::new([f1, f2]);
//!
//! // Values and the 2×2 Jacobian in one DualN sweep, then J·Δx = −f.
//! let x0 = [2.02, 0.98];
//! let (f, j) = sys.eval_jacobian(&x0);
//! let dx = j.solve(&Vector::new([-f[0], -f[1]])).unwrap();
//! let x1 = [x0[0] + dx[0], x0[1] + dx[1]];
//!
//! // One Newton step contracts the residual quadratically.
//! let r0 = f[0].abs() + f[1].abs();
//! let r1 = sys.eval(&x1);
//! assert!(r1[0].abs() + r1[1].abs() < 0.05 * r0);
//! ```
//!
//! # Feature flags
//!
//! - `std` (default): float math via the standard library.
//! - `libm`: float math via the `libm` crate for `no_std` targets (if both
//!   are enabled, `std` wins).
//! - `specialization` (nightly only): dispatch fixed-size matrix-multiply
//!   kernels via `min_specialization`.

#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(feature = "specialization", feature(min_specialization))]
#![warn(missing_docs)]
#![forbid(unsafe_code)]

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
