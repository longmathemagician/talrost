//! Polyhedral homotopy continuation (Huber–Sturmfels).
//!
//! A polyhedral homotopy solves a sparse polynomial system
//! `F = (f_1, …, f_n)` in `n` variables by tracking paths from the roots of
//! easy *binomial start systems*, one per **mixed cell** of the subdivision
//! induced by a generic lifting of the supports (Huber & Sturmfels, *A
//! polyhedral method for solving sparse polynomial systems*, Math. Comp. 64,
//! 1995). Both halves of that pipeline live here:
//!
//! **Offline** — depends only on the monomial structure (plus, for the
//! start roots, the coefficient values):
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
//! **Online** — tracks each start root to a root of the target:
//!
//! 5. [`CellHomotopy`]: the cell's homotopy `H_i(y, t) = Σ c·t^e·y^a`,
//!    which is the binomial system at `t = 0` and the target at `t = 1`;
//!    [`track_path`] follows one root with an Euler predictor and Newton
//!    corrector under [`TrackOptions`], reporting a [`PathResult`].
//! 6. [`solve`]: the end-to-end driver — every start of every cell,
//!    collected into a [`SolveReport`] with the mixed volume and
//!    deduplication helpers.
//!
//! The enumeration and driver allocate (`Vec`): cell counts are runtime
//! values, so the module is `std`-only, unlike the stack-only tower it
//! builds on (the tracker itself is allocation-free by construction).
//!
//! # Example
//!
//! The sparse pair `A_1 = {1, x, xy}`, `A_2 = {1, y, xy}` has mixed volume
//! 2 — below its Bézout bound of 4, the polyhedral advantage — so exactly
//! two paths are tracked, and both land on genuine roots:
//!
//! ```
//! use talrost::complex::c64;
//! use talrost::mvpoly::{MPoly, MSystem, Monomial};
//! use talrost::solvers::homotopy::{solve, PathStatus, TrackOptions};
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
//! let report = solve(&system, 42, &TrackOptions::default()).expect("generic lifting");
//! assert_eq!(report.mixed_volume, 2);
//! assert!(report.paths.iter().all(|p| p.status == PathStatus::Converged));
//! for root in report.solutions() {
//!     let residual = system.eval(&root);
//!     assert!(residual.iter().all(|r| r.magnitude() < 1e-8));
//! }
//! assert_eq!(report.distinct_solutions(1e-6).len(), 2);
//! ```

pub mod cells;
pub mod driver;
pub mod start;
pub mod support;
pub mod track;

pub use cells::{mixed_cells, mixed_volume, GenericityError, MixedCell};
pub use driver::{solve, SolveReport};
pub use start::{binomial_solutions, start_solutions};
pub use support::{random_liftings, Lifting, Support};
pub use track::{track_path, CellHomotopy, PathResult, PathStatus, TrackOptions};
