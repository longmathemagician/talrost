//! Polyhedral homotopy continuation (Huber–Sturmfels) — the **offline**
//! phase.
//!
//! A polyhedral homotopy solves a sparse polynomial system
//! `F = (f_1, …, f_n)` in `n` variables by tracking paths from the roots of
//! easy *binomial start systems*, one per **mixed cell** of the subdivision
//! induced by a generic lifting of the supports (Huber & Sturmfels, *A
//! polyhedral method for solving sparse polynomial systems*, Math. Comp. 64,
//! 1995). This module implements the offline half of that pipeline — the
//! part that depends only on the monomial structure (plus, for the start
//! roots, the coefficient values):
//!
//! 1. [`Support`]: the exponent-vector set of each equation
//!    ([`Support::from_msystem`]).
//! 2. [`Lifting`]: generic real lift values, one per support point
//!    ([`Lifting::random`] / [`random_liftings`] — reproducible by seed).
//! 3. [`mixed_cells`]: enumerate the *fine* mixed cells of the induced
//!    subdivision; [`mixed_volume`] sums their exact `|det V|` volumes —
//!    Bernstein's generic root count on the torus `(ℂ*)ⁿ`.
//! 4. [`start_solutions`]: solve each cell's binomial system `x^V = β` in
//!    closed form through the Smith normal form of its edge matrix
//!    ([`crate::lattice::smith_normal_form`]).
//!
//! The online half — tracking `H(x, t)` from these start roots to the roots
//! of the target system — is deliberately **not** here (it is Phase 8 of the
//! roadmap). Everything in this module may allocate (`Vec`): cell counts are
//! runtime values, so the module is `std`-only, unlike the stack-only tower
//! it builds on.
//!
//! # Example
//!
//! The sparse pair `A_1 = {1, x, xy}`, `A_2 = {1, y, xy}` has mixed volume
//! 2 — below its Bézout bound of 4, the polyhedral advantage — so exactly
//! two start solutions are produced across all cells:
//!
//! ```
//! use talrost::complex::c64;
//! use talrost::mvpoly::{MPoly, MSystem, Monomial};
//! use talrost::solvers::homotopy::{mixed_cells, random_liftings, start_solutions, Support};
//!
//! let c = c64::new;
//! let f1 = MPoly::new(
//!     [c(2.0, 1.0), c(3.0, -0.5), c(-1.0, 2.0)],
//!     [Monomial::new([0, 0]), Monomial::new([1, 0]), Monomial::new([1, 1])],
//! );
//! let f2 = MPoly::new(
//!     [c(1.0, -1.0), c(-2.0, 0.5), c(5.0, 1.5)],
//!     [Monomial::new([0, 0]), Monomial::new([0, 1]), Monomial::new([1, 1])],
//! );
//! let system = MSystem::new([f1, f2]);
//!
//! let supports = Support::from_msystem(&system);
//! let liftings = random_liftings::<f64, 2>(&supports, 42);
//! let cells = mixed_cells(&supports, &liftings).expect("re-lift with a new seed");
//!
//! let starts: usize = cells.iter().map(|cell| start_solutions(&system, cell).len()).sum();
//! assert_eq!(starts, 2);
//! ```

pub mod cells;
pub mod start;
pub mod support;

pub use cells::{mixed_cells, mixed_volume, GenericityError, MixedCell};
pub use start::{binomial_solutions, start_solutions};
pub use support::{random_liftings, Lifting, Support};
