//! Univariate polynomial root finders.
//!
//! Each module keeps its solver's *native* output convention (root order,
//! `NAN`/infinity sentinels for missing roots); the ergonomic entry points
//! are the per-degree `roots(tol)` methods on
//! [`crate::polynomial::Polynomial`], which pack these raw outputs into a
//! counted, ascending [`crate::roots::Roots`].

// pub mod autodiff;
// pub mod orellana;
// pub mod newton;
/// Cem Yuksel's numeric root finder (HPG 2022): cubics and quartics by
/// recursive derivative root-bracketing plus safeguarded Newton iteration.
pub mod yuksel;
// pub mod homotopy_tests;
// pub mod phc;
/// Jim Blinn's homogeneous closed-form quadratic/cubic solvers, formulated
/// to avoid catastrophic cancellation between the two roots.
pub mod blinn;
