//! LU factorization with partial pivoting — the crate's linear-solve
//! primitive.
//!
//! The Newton corrector of a path tracker solves `J·Δx = −H` every
//! iteration; `inverse()` is the wrong primitive for that. Factor once with
//! [`super::Matrix::lu`], then run [`Lu::solve`] per right-hand side.

use crate::algebra::{Monoid, Semiring};
use crate::real::Real;
use crate::scalar::Scalar;
use crate::vector::Vector;

use super::Matrix;

/// A packed LU factorization `P·A = L·U` of a square matrix, produced by
/// [`super::Matrix::lu`].
///
/// `lu` stores `L` (unit lower triangle, implicit ones) below the diagonal
/// and `U` on/above it; `perm` records the row permutation `P` (position `k`
/// holds the original row index) and `odd` its parity (which flips the
/// determinant's sign).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Lu<T: Scalar, const N: usize> {
    lu: [[T; N]; N],
    perm: [usize; N],
    odd: bool,
}

impl<T: Scalar, const N: usize> Lu<T, N> {
    /// Doolittle factorization with partial pivoting by `norm_sqr` (cheap,
    /// exact, and meaningful for complex scalars where `PartialOrd` is not).
    /// Returns `None` when no nonzero pivot remains — the matrix is singular
    /// to working precision.
    pub(super) fn factor(m: &Matrix<T, N, N>) -> Option<Self> {
        let mut a = m.e;
        let mut perm = [0usize; N];
        for (k, p) in perm.iter_mut().enumerate() {
            *p = k;
        }
        let mut odd = false;

        for k in 0..N {
            // Select the remaining row whose k-th entry has the largest
            // squared norm.
            let mut pivot = k;
            let mut best = a[k][k].norm_sqr();
            for (r, row) in a.iter().enumerate().skip(k + 1) {
                let v = row[k].norm_sqr();
                if v > best {
                    best = v;
                    pivot = r;
                }
            }
            if best == T::Real::ZERO {
                return None; // singular
            }
            if pivot != k {
                a.swap(k, pivot);
                perm.swap(k, pivot);
                odd = !odd;
            }
            // The pivot row is a `Copy` array: snapshot it so the rows below
            // can be mutably iterated without aliasing.
            let pivot_row = a[k];
            for row in a.iter_mut().skip(k + 1) {
                let factor = row[k] / pivot_row[k];
                row[k] = factor; // store the L multiplier in the zeroed slot
                for (v, &p) in row.iter_mut().zip(pivot_row.iter()).skip(k + 1) {
                    *v -= factor * p;
                }
            }
        }

        Some(Self { lu: a, perm, odd })
    }

    /// Solves `A·x = b` by permuting `b`, then forward substitution through
    /// the unit-lower `L` and back substitution through `U`. Infallible: a
    /// successfully factored matrix always solves.
    pub fn solve(&self, b: &Vector<T, N>) -> Vector<T, N> {
        // y = P·b
        let mut y = [T::ZERO; N];
        for (yi, &p) in y.iter_mut().zip(self.perm.iter()) {
            *yi = b.b[p];
        }
        // L·z = y (unit diagonal, so no division)
        for i in 0..N {
            let mut acc = T::ZERO;
            for (l, &yj) in self.lu[i][..i].iter().zip(y[..i].iter()) {
                acc += *l * yj;
            }
            y[i] -= acc;
        }
        // U·x = z
        for i in (0..N).rev() {
            let mut acc = T::ZERO;
            for (l, &yj) in self.lu[i][(i + 1)..].iter().zip(y[(i + 1)..].iter()) {
                acc += *l * yj;
            }
            y[i] -= acc;
            y[i] /= self.lu[i][i];
        }
        Vector::new(y)
    }

    /// The ratio of the smallest to the largest pivot norm,
    /// `min_i |U_ii| / max_i |U_ii|`, in `(0, 1]` (a successful
    /// factorization has no zero pivot; `1` exactly when all pivots share
    /// one norm, e.g. the identity).
    ///
    /// This is a **cheap singularity-proximity signal, not a condition
    /// number**: partial pivoting makes a tiny trailing pivot the usual
    /// symptom of near-singularity, so a small ratio flags trouble for
    /// free from the already-computed factorization — but a matrix can be
    /// ill-conditioned with unsuspicious pivots (and vice versa), so treat
    /// the ratio as a hint, never a bound. Path trackers store it as the
    /// [`crate::solvers::homotopy::PathResult::pivot_ratio`] diagnostic.
    pub fn pivot_ratio(&self) -> T::Real {
        let mut min = T::Real::INFINITY;
        let mut max = T::Real::ZERO;
        for (k, row) in self.lu.iter().enumerate() {
            let p = row[k].norm_sqr();
            min = min.min(p);
            max = max.max(p);
        }
        if N == 0 {
            return T::Real::ONE; // no pivots: vacuously perfect
        }
        // Ratio of squared norms, then one square root: same value as
        // min |U_ii| / max |U_ii| with half the sqrt calls.
        (min / max).sqrt()
    }

    /// The determinant of the factored matrix: the product of `U`'s diagonal,
    /// negated when the row permutation is odd.
    pub fn determinant(&self) -> T {
        let mut det = T::ONE;
        for (k, row) in self.lu.iter().enumerate() {
            det *= row[k];
        }
        if self.odd {
            -det
        } else {
            det
        }
    }
}
